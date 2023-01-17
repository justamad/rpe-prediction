from .base_loader import BaseSubjectLoader, LoadingException
from src.camera import fuse_cameras
from os.path import join, exists, isdir

import pandas as pd
import os
import pathlib


class AzureSubjectLoader(BaseSubjectLoader):

    def __init__(self, root_path: str, subject_name: str):
        super().__init__(subject_name)
        self._azure_path = join(root_path, "azure")
        if not exists(self._azure_path):
            raise LoadingException(f"Azure file not present in {self._azure_path}")

        # Load data into two dictionaries: {trial_n: abs_path, ...}
        all_trials = list(map(lambda x: join(self._azure_path, x), os.listdir(self._azure_path)))
        all_trials = list(filter(lambda x: isdir(x), all_trials))

        sub_trials = list(filter(lambda x: "sub" in x, all_trials))
        master_trials = list(filter(lambda x: "master" in x, all_trials))

        self._sub_trials = {int(pathlib.PurePath(v).name.split("_")[0]) - 1: v for v in sub_trials}
        self._master_trials = {int(pathlib.PurePath(v).name.split("_")[0]) - 1: v for v in master_trials}

    def get_trial_by_set_nr(self, trial_nr: int):
        if trial_nr not in self._sub_trials or trial_nr not in self._master_trials:
            raise LoadingException(f"{str(self)}: Error when loading set: {trial_nr} for subject {self._azure_path}")

        sub = join(self._sub_trials[trial_nr], "positions_3d.csv")
        master = join(self._master_trials[trial_nr], "positions_3d.csv")

        master_df = pd.read_csv(master, index_col=0, sep=";")
        sub_df = pd.read_csv(sub, index_col=0, sep=";")
        fused_df = fuse_cameras(master_df, sub_df)
        return fused_df

    def get_nr_of_sets(self):
        return min(len(self._sub_trials), len(self._master_trials))

    def __repr__(self):
        return f"StereoAzureLoader {self._subject_name}"
