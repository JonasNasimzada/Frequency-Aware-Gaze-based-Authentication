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

        self.samples = samples
        self.metadata = pd.DataFrame(metadata)

        subjects_tensor = torch.tensor(subjects, dtype=torch.long)
        unique_subjects = subjects_tensor.unique()
        self.classes = torch.bucketize(subjects_tensor, unique_subjects)  # class indices
        self.n_classes = len(unique_subjects)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        x = self.samples[index]

        y_dict = self.metadata.iloc[index].to_dict()
        y_dict["class"] = self.classes[index].item()
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
