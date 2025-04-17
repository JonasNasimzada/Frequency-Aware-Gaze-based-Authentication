# This work is licensed under a "Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License"
# (https://creativecommons.org/licenses/by-nc-sa/4.0/).
#
# Author: Dillon Lohr (djl70@txstate.edu)
# Property of Texas State University.

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TASK_TO_NUM = {
    "HSS": 0,
    "RAN": 1,
    "TEX": 2,
    "FXS": 3,
    "VD1": 4,
    "VD2": 5,
    "BLG": 6,
}


class SubsequenceDataset(Dataset):
    def __init__(
        self,
        sequences: List[np.ndarray],
        labels: pd.DataFrame,
        subsequence_length: int,
        mn: Optional[float] = None,
        sd: Optional[float] = None,
    ):
        super().__init__()

        samples = []
        subjects = []
        metadata = {
            k: []
            for k in (
                "nb_round",
                "nb_subject",
                "nb_session",
                "nb_task",
                "nb_subsequence",
                "exclude",
            )
        }
        #for recording, (_, label) in zip(sequences, labels.iterrows()):
        #    nb_round = int(label["nb_round"])
        #    nb_subject = int(label["nb_subject"])
        #    nb_session = int(label["nb_session"])
        #    nb_task = TASK_TO_NUM[label["task"]]
        #    #print(f"before = {recording.shape}")
        #    # Extract fixed-length, non-overlapping subsequences
        #    recording_tensor = torch.from_numpy(recording).float()
        #    #subsequences = recording_tensor.unfold(
        #    #    dimension=-1, size=subsequence_length, step=subsequence_length
        #    #)
        #    #subsequences = recording_tensor.swapdims(0, 1)  # (batch, feature, seq)
        #    #print(f"after = {recording_tensor.shape}")
        #    n_seq = recording_tensor.size(0)
        #    nb_subsequence = np.arange(n_seq)
        #    portion_nan = recording_tensor.isnan().any(dim=1).float().mean(dim=-1)
        #    exclude = portion_nan > 0.5
#
        #    samples.append(recording_tensor)
        #    subjects.append(torch.LongTensor([nb_subject] * n_seq))
        #    metadata["nb_round"].extend([nb_round] * n_seq)
        #    metadata["nb_subject"].extend([nb_subject] * n_seq)
        #    metadata["nb_session"].extend([nb_session] * n_seq)
        #    metadata["nb_task"].extend([nb_task] * n_seq)
        #    metadata["nb_subsequence"].extend(nb_subsequence)
        #    metadata["exclude"].extend(exclude.numpy())

        for recording, (_, label) in zip(sequences, labels.iterrows()):
            recording_tensor = torch.from_numpy(recording).float()

            nb_round = int(label["nb_round"])
            nb_subject = int(label["nb_subject"])
            nb_session = int(label["nb_session"])
            nb_task = TASK_TO_NUM[label["task"]]
        
            # 1. Append entire recording as one sample
            samples.append(recording_tensor)
        
            # 2. Append exactly one line of metadata per recording
            subjects.append(torch.LongTensor([nb_subject]))
            metadata["nb_round"].append(int(label["nb_round"]))
            metadata["nb_subject"].append(int(label["nb_subject"]))
            metadata["nb_session"].append(int(label["nb_session"]))
            metadata["nb_task"].append(TASK_TO_NUM[label["task"]])
            metadata["nb_subsequence"].append(0)  # or some aggregate statistic
            metadata["exclude"].append(0)        # or an aggregate

        #self.samples = torch.cat(samples, dim=0)
        self.samples = samples
        self.metadata = pd.DataFrame(metadata)

        #subjects = torch.cat(subjects, dim=0)
        #unique_subjects = subjects.unique()
        #self.classes = torch.bucketize(subjects, unique_subjects)
        #self.n_classes = len(unique_subjects)
        
        subjects_tensor = torch.tensor(subjects, dtype=torch.long)
        unique_subjects = subjects_tensor.unique()
        self.classes = torch.bucketize(subjects_tensor, unique_subjects)  # class indices
        self.n_classes = len(unique_subjects)

        #if mn is None or sd is None:
        #    #x = torch.clamp(self.samples, min=-1000.0, max=1000.0).numpy()
        #    mn = np.nanmean(x)
        #    sd = np.nanstd(x)
        #self.mn = mn
        #self.sd = sd

    def __len__(self) -> int:
        #print(f"--len-- : {len(self.samples)}")
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        #print(f"length list {len(self.samples)}")
        #print(f"index {index}")
        x = self.samples[index]
       # x = torch.clamp(x, min=-1000.0, max=1000.0)
        #x = (x - self.mn) / self.sd
        #x = torch.nan_to_num(x, nan=0.0)

        y_dict = self.metadata.iloc[index].to_dict()
        y_dict["class"] = self.classes[index].item()
        #y = torch.LongTensor(
        #    [
        #        y[k]
        #        for k in (
        #            "class",
        #            "nb_round",
        #            "nb_subject",
        #            "nb_session",
        #            "nb_task",
        #            "nb_subsequence",
        #            "exclude",
        #        )
        #    ]
        #)
        y_tensor = torch.LongTensor([
            y_dict["class"],
            y_dict["nb_round"],
            y_dict["nb_subject"],
            y_dict["nb_session"],
            y_dict["nb_task"],
            y_dict["nb_subsequence"],
            y_dict["exclude"]
        ])

        return x, y_tensor
