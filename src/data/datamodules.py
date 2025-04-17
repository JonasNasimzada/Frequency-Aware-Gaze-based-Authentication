# This work is licensed under a "Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License"
# (https://creativecommons.org/licenses/by-nc-sa/4.0/).
#
# Author: Dillon Lohr (djl70@txstate.edu)
# Property of Texas State University.

import os
import pickle
import re
import shutil
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import (
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
)
from pytorch_metric_learning import samplers
from scipy.signal import savgol_filter, spectrogram
from torch.utils.data import DataLoader
from tqdm import tqdm

from .assign_groups import assign_groups
from .datasets import SubsequenceDataset
from .download import download
from .downsample import downsample_recording
from .zip import extract

# https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257/3
GAZEBASE_URL = "https://ndownloader.figshare.com/files/27039812"

# https://osf.io/5zpvk/
JUDO1000_URL = "https://osf.io/4wy7s/download"


class GazeBaseDataModule(pl.LightningDataModule):
    def __init__(
            self,
            current_fold: int = 0,
            base_dir: str = "./data/gazebase_v3",
            downsample_factors: Sequence[int] = tuple(),
            subsequence_length_before_downsampling: int = 5000,
            classes_per_batch: int = 16,
            samples_per_class: int = 16,
            compute_map_at_r: bool = False,
            batch_size_for_testing: Optional[int] = None,
            noise_sd: Optional[float] = None,
    ):
        super().__init__()

        self.initial_sampling_rate_hz = 1000
        self.downsample_factors = downsample_factors
        self.total_downsample_factor = np.prod(self.downsample_factors)
        self.noise_sd = noise_sd

        self.subsequence_length = int(
            subsequence_length_before_downsampling
            // self.total_downsample_factor
        )

        self.base_dir = Path(base_dir)
        self.archive_path = self.base_dir / "gazebase.zip"
        self.raw_file_dir = self.base_dir / "raw"
        self.tmp_file_dir = self.base_dir / "tmp"
        self.processed_path = (
                self.base_dir
                / "processed"
                / (
                        f"gazebase_savgol_ds{int(self.total_downsample_factor)}"
                        + f"_{'normal' if self.noise_sd is None else 'degraded'}.pkl"
                )
        )

        self.current_fold = current_fold
        self.n_folds = 4
        self.nb_round_for_test_subjects = 6

        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.fit_batch_size = self.classes_per_batch * self.samples_per_class
        self.test_batch_size = batch_size_for_testing or self.fit_batch_size

        self.compute_map_at_r = compute_map_at_r

        self.train_loader: DataLoader
        self.val_loaders: List[DataLoader]
        self.test_loaders: List[DataLoader]

        self.n_classes: int
        self.zscore_np: float
        self.zscore_sd: float

    def prepare_data(self) -> None:
        self.download_and_process_gazebase()
        # if not self.processed_path.exists():
        #   self.download_and_process_gazebase()

    def load_img(self, x):
        image = cv2.imread(x)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(image, (224, 224))
        return resized_image

    def setup(self, stage: Optional[str] = None) -> None:
        """
                Loads the processed .pkl file, then sets up:
                  - self.train_loader
                  - self.val_loaders
                  - self.test_loaders
                """
        if stage == "test":
            batch_size = self.test_batch_size
        else:
            batch_size = self.fit_batch_size

        fold_label = f"fold{self.current_fold}"
        with open(self.processed_path, "rb") as f:
            data_dict = pickle.load(f)
            # print([v["inputs"] for k, v in data_dict.items()])

        # ------------------------------------------------
        # TRAIN SPLIT: all folds except self.current_fold + no test
        # ------------------------------------------------

        train_X = [
            self.load_img(x).T  # shape: (feature, seq)
            for split, v in data_dict.items()
            if split not in (fold_label, "test")
            for x in v["inputs"]
        ]
        train_y = pd.concat(
            [
                v["labels"]
                for split, v in data_dict.items()
                if split not in (fold_label, "test")
            ],
            ignore_index=True,
            axis=0,
        )

        # Remove BLG data from train set
        is_train_blg = train_y["task"].str.fullmatch("BLG")
        train_X = [x for x, is_blg in zip(train_X, is_train_blg) if not is_blg]
        train_y = train_y.loc[~is_train_blg, :]

        train_set = SubsequenceDataset(
            train_X, train_y, self.subsequence_length, mn=None, sd=None
        )
        # None = train_set.mn
        # None = train_set.sd
        self.n_classes = train_set.n_classes

        train_sampler = samplers.MPerClassSampler(
            train_set.classes,
            self.samples_per_class,
            batch_size=batch_size,
            length_before_new_iter=len(train_set),
        )
        self.train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=4,
        )

        # ------------------------------------------------
        # VAL SPLIT
        # ------------------------------------------------
        val_X = [self.load_img(x).T for x in data_dict[fold_label]["inputs"]]
        val_y = data_dict[fold_label]["labels"]
        self.val_loaders = []

        if stage != "test":
            # Remove BLG data from val set during training
            is_val_blg = val_y["task"].str.fullmatch("BLG")
            val_X = [x for x, is_blg in zip(val_X, is_val_blg) if not is_blg]
            val_y = val_y.loc[~is_val_blg, :]

        full_val_set = SubsequenceDataset(
            val_X,
            val_y,
            self.subsequence_length,
            mn=0.0,
            sd=None,
        )

        if stage == "test":
            full_val_sampler = None
        else:
            # MPerClassSampler used for computing multi-similarity loss
            full_val_sampler = samplers.MPerClassSampler(
                full_val_set.classes,
                self.samples_per_class,
                batch_size=batch_size,
                length_before_new_iter=len(full_val_set),
            )

        full_val_loader = DataLoader(
            full_val_set,
            batch_size=batch_size,
            shuffle=False,
            sampler=full_val_sampler,
            num_workers=4,
        )
        self.val_loaders.append(full_val_loader)

        # Optionally add more val loaders if MAP@R is enabled
        if stage != "test" and self.compute_map_at_r:
            val_is_tex = val_y["task"].str.fullmatch("TEX")
            val_tex_X = [x for x, is_tex in zip(val_X, val_is_tex) if is_tex]
            val_tex_y = val_y.loc[val_is_tex, :]
            val_tex_set = SubsequenceDataset(
                val_tex_X,
                val_tex_y,
                self.subsequence_length,
                mn=None,
                sd=None,
            )
            val_tex_loader = DataLoader(
                val_tex_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
            )
            self.val_loaders.append(val_tex_loader)

            # Also gather TEX from train set for MAP@R
            train_is_tex = train_y["task"].str.fullmatch("TEX")
            train_tex_X = [x for x, is_tex in zip(train_X, train_is_tex) if is_tex]
            train_tex_y = train_y.loc[train_is_tex, :]
            train_tex_set = SubsequenceDataset(
                train_tex_X,
                train_tex_y,
                self.subsequence_length,
                mn=None,
                sd=None,
            )
            train_tex_loader = DataLoader(
                train_tex_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
            )
            self.val_loaders.append(train_tex_loader)

        # ------------------------------------------------
        # TEST SPLIT
        # ------------------------------------------------
        self.test_loaders = []
        if stage == "test":
            test_X = [self.load_img(x).T for x in data_dict["test"]["inputs"]]
            test_y = data_dict["test"]["labels"]
            test_set = SubsequenceDataset(
                test_X,
                test_y,
                self.subsequence_length,
                mn=None,
                sd=None,
            )
            test_loader = DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
            )
            self.test_loaders.append(test_loader)

            # Also append the "full_val_loader" for test-time evaluation
            self.test_loaders.append(full_val_loader)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_loaders

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_loaders

    def download_and_process_gazebase(self) -> None:
        # Download and extract GazeBase archives if necessary
        if not self.raw_file_dir.exists() or self.tmp_file_dir.exists():
            self.archive_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.archive_path.exists():
                print("Downloading GazeBase from figshare")
                download(GAZEBASE_URL, self.archive_path)

            # If the temporary directory still exists, there must have
            # been an error when extracting files. Delete old directories to start fresh.
            if self.tmp_file_dir.exists():
                print("Removing old directories")
                shutil.rmtree(self.tmp_file_dir)
                if self.raw_file_dir.exists():
                    shutil.rmtree(self.raw_file_dir)

            print("Extracting GazeBase archive to temporary directory")
            self.tmp_file_dir.mkdir(parents=True, exist_ok=True)
            extract(self.archive_path, self.tmp_file_dir)

            print("Extracting subject archives to temporary directory")
            subject_archives = list(self.tmp_file_dir.rglob("Subject_*.zip"))
            for archive in tqdm(subject_archives):
                extract(archive, self.tmp_file_dir)

            print("Moving data files out of temporary directory")
            self.raw_file_dir.mkdir(parents=True, exist_ok=True)
            data_files = list(self.tmp_file_dir.rglob("S_*.csv"))
            for file in tqdm(data_files):
                new_file_path = self.raw_file_dir / file.name
                shutil.move(file, new_file_path)

            print("Deleting temporary directory")
            shutil.rmtree(self.tmp_file_dir)
        # Process recordings
        print("HEERE")
        filename_pattern = r"S_(\d)(\d+)_S(\d)_(\w+)"
        recording_paths = sorted(list(self.raw_file_dir.iterdir()))
        inputs = []
        labels = []

        print("Processing all recordings")
        for path in tqdm(recording_paths):
            # Read the CSV
            df = pd.read_csv(path)

            if 'x' not in df.columns or 'y' not in df.columns:
                raise ValueError(f"CSV file {path} must contain 'x' and 'y' columns.")

            # Downsample gaze data
            gaze, sampling_rate = downsample_recording(
                df, self.downsample_factors, self.initial_sampling_rate_hz
            )

            # Optionally add noise
            if self.noise_sd is not None:
                noise = np.random.randn(*gaze.shape) * self.noise_sd
                gaze += noise

            # Interpolate any missing data
            gaze = pd.DataFrame(gaze).interpolate(method="linear", limit_direction="both").values

            # Savitzky-Golay filter for velocity
            vel = savgol_filter(gaze, window_length=7, polyorder=2, deriv=1, axis=0, mode="nearest")
            vel *= sampling_rate  # deg/sec

            # ----- SPECTROGRAM LOGIC START -----
            # Compute magnitude of velocity
            vel_magnitude = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2)

            # Compute spectrogram for combined velocity magnitude
            nperseg = 256
            noverlap = nperseg // 2
            fs = sampling_rate

            f, t, Sxx = spectrogram(
                vel_magnitude, fs, nperseg=nperseg, noverlap=noverlap, scaling='density'
            )
            inputs.append(Sxx.astype(np.float32))

            # Create a plots directory
            plot_dir = 'data/spectrogram_plots'
            os.makedirs(plot_dir, exist_ok=True)

            # Plot and save spectrogram
            plt.figure(figsize=(12, 6), frameon=False)
            plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-20), shading='gouraud')
            plt.savefig(os.path.join(
                plot_dir,
                f"{os.path.splitext(os.path.basename(path))[0]}_vel_combined.png"
            ))
            plt.close()
            # ----- SPECTROGRAM LOGIC END -----
            # Parse labels from filename
            pattern_match = re.match(filename_pattern, path.stem)
            match_groups = pattern_match.groups()
            label = {
                "nb_round": match_groups[0],
                "nb_subject": match_groups[1],
                "nb_session": match_groups[2],
                "task": match_groups[3],
            }
            labels.append(label)

        labels_df = pd.DataFrame(labels)

        # The test set contains all the data from subjects present in Round 6
        print("Creating held-out test set")
        subjects = labels_df.loc[:, "nb_subject"]
        nb_round_as_int = labels_df.loc[:, "nb_round"].astype(int)
        is_test_round = nb_round_as_int == self.nb_round_for_test_subjects
        subjects_in_test_set = subjects[is_test_round].unique()
        is_subject_in_test_set = subjects.isin(subjects_in_test_set)

        n_subjects = len(subjects.unique())
        test_pct_subjects = 100 * len(subjects_in_test_set) / n_subjects
        test_pct_recordings = 100 * is_subject_in_test_set.mean()
        print(
            f"Created test set with {test_pct_subjects:.2f}% of subjects"
            + f", {test_pct_recordings:.2f}% of recordings"
        )

        print("Assigning remaining subjects to folds")
        leftover_subjects = subjects[~is_subject_in_test_set]
        leftover_unique = leftover_subjects.unique()

        # Weight each subject by the number of recordings they have.
        # (heapq usage omitted here but typically you'd do something similar)
        weights = [-np.sum(leftover_subjects == s) for s in leftover_unique]
        fold_to_id, grp = assign_groups(self.n_folds, leftover_unique, weights)

        # Verify that the folds are roughly balanced
        least, greatest = np.min(grp, axis=0), np.max(grp, axis=0)
        subject_diff = greatest[0] - least[0]
        recording_diff = greatest[1] - least[1]
        print(f"Max - min of # subjects in each fold: {subject_diff}")
        print(f"Max - min of # recordings in each fold: {recording_diff}")

        # Create a dictionary of inputs and labels for each data split
        def get_split_data(split_subjects):
            split_indices = np.where(subjects.isin(split_subjects))[0]
            return {
                "inputs": [inputs[i] for i in split_indices],
                "labels": labels_df.iloc[split_indices, :],
            }

        test_idx = {"test": get_split_data(subjects_in_test_set)}
        fold_idx = {
            f"fold{k}": get_split_data(v) for k, v in fold_to_id.items()
        }
        data_dict = {**test_idx, **fold_idx}

        # Save dictionary of processed data
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.processed_path, "wb") as f:
            pickle.dump(data_dict, f)
        print(f"Finished processing data. Saved to '{self.processed_path}'.")


class JuDo1000DataModule(pl.LightningDataModule):
    def __init__(
            self,
            zscore_mn: float,
            zscore_sd: float,
            base_dir: str = "./data/judo1000",
            subsequence_length: int = 5000,
            batch_size: int = 256,
    ):
        super().__init__()

        # None = zscore_mn
        # None = zscore_sd

        self.subsequence_length = subsequence_length

        self.base_dir = Path(base_dir)
        self.archive_path = self.base_dir / "judo1000.zip"
        self.raw_file_dir = self.base_dir / "raw"
        self.tmp_file_dir = self.base_dir / "tmp"
        self.processed_path = (
                self.base_dir / "processed" / "judo1000_savgol_ds1_normal.pkl"
        )

        self.batch_size = batch_size

        self.test_loader: DataLoader

    def prepare_data(self) -> None:
        if not self.processed_path.exists():
            self.download_and_process_judo1000()

    def setup(self, stage: Optional[str] = None) -> None:
        with open(self.processed_path, "rb") as f:
            data_dict = pickle.load(f)

        test_X = [x.T for x in data_dict["test"]["inputs"]]
        test_y = data_dict["test"]["labels"]
        test_set = SubsequenceDataset(
            test_X,
            test_y,
            self.subsequence_length,
            mn=None,
            sd=None,
        )
        self.test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_loader

    def download_and_process_judo1000(self) -> None:
        # Download and extract JuDo1000 archive if necessary
        if not self.raw_file_dir.exists() or self.tmp_file_dir.exists():
            self.archive_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.archive_path.exists():
                print("Downloading JuDo1000 from OSF")
                download(JUDO1000_URL, self.archive_path)

            # If the temporary directory still exists, there must have
            # been an error when extracting files.  Delete the old
            # directories to start fresh.
            if self.tmp_file_dir.exists():
                print("Removing old directories")
                shutil.rmtree(self.tmp_file_dir)
                if self.raw_file_dir.exists():
                    shutil.rmtree(self.raw_file_dir)

            print("Extracting JuDo1000 archive to temporary directory")
            self.tmp_file_dir.mkdir(parents=True, exist_ok=True)
            extract(self.archive_path, self.tmp_file_dir)

            print("Moving data files out of temporary directory")
            self.raw_file_dir.mkdir(parents=True, exist_ok=True)
            data_files = list(self.tmp_file_dir.rglob("*.csv"))
            for file in tqdm(data_files):
                new_file_path = self.raw_file_dir / file.name
                shutil.move(file, new_file_path)

            print("Deleting temporary directory")
            shutil.rmtree(self.tmp_file_dir)

        screen_px = np.array([1280, 1024])
        screen_mm = np.array([380, 300])
        distance_mm = 680
        invert_direction = np.array([False, True])  # for px, +Y is down

        def pixels_to_degrees(gaze_px):
            mm2px = screen_px / screen_mm
            sign_scalar = -2 * invert_direction + 1
            gaze_rad = sign_scalar * (gaze_px - screen_px * 0.5)
            gaze_deg = np.rad2deg(np.arctan2(gaze_rad, mm2px * distance_mm))
            return gaze_deg

        # Process recordings
        recording_paths = sorted(
            [
                f
                for f in self.raw_file_dir.iterdir()
                if not f.stem.endswith("_TrialVars")
            ]
        )
        inputs = []
        labels = []
        print("Processing all recordings")
        for data_path in tqdm(recording_paths):
            vars_path = data_path.with_name(data_path.stem + "_TrialVars.csv")

            # Select trials with the largest grid and duration
            vars_df = pd.read_csv(vars_path, sep="\t")
            grid_match = vars_df["grid"] == 0.25
            dur_match = vars_df["dur"] == 1000
            desired_config = grid_match & dur_match
            trials = vars_df.loc[desired_config, "trialId"].to_numpy()

            data_df = pd.read_csv(data_path, sep="\t")
            recording = []
            for trial in trials:
                is_trial = data_df["trialId"] == trial
                trial_df = data_df.loc[is_trial, :]
                x = trial_df[["x_left", "x_right"]].mean(axis=1).to_numpy()
                y = trial_df[["y_left", "y_right"]].mean(axis=1).to_numpy()
                gaze_px = np.stack([x, y], axis=1)

                assert (
                        gaze_px.shape[0] >= self.subsequence_length
                ), "Trial too short"
                gaze_px = gaze_px[: self.subsequence_length]

                gaze_deg = pixels_to_degrees(gaze_px)
                vel = savgol_filter(
                    gaze_deg, 7, 2, deriv=1, axis=0, mode="nearest"
                )
                vel *= 1000.0  # deg/sec
                recording.append(vel.astype(np.float32))
            # The "recording" will be split back into its subsequences
            # (i.e., trials) within the Dataset
            recording = np.concatenate(recording, axis=0)
            inputs.append(recording)

            subject, session = data_path.stem.split("_")
            label = {
                "nb_round": "1",
                "nb_subject": subject,
                "nb_session": session,
                "task": "RAN",
            }
            labels.append(label)
        labels_df = pd.DataFrame(labels)

        # Save dictionary of processed data
        data_dict = {"test": {"inputs": inputs, "labels": labels_df}}
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.processed_path, "wb") as f:
            pickle.dump(data_dict, f)
        print(f"Finished processing data. Saved to '{self.processed_path}'.")
