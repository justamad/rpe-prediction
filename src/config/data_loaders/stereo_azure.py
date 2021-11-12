from .base_loader import BaseSubjectLoader, LoadingException
from os.path import join, exists, isdir

import os


class StereoAzureSubjectLoader(BaseSubjectLoader):

    def __init__(self, root_path: str, subject: str):
        super().__init__()
        self._azure_path = join(root_path, "azure")
        if not exists(self._azure_path):
            raise LoadingException(f"Azure file not present in {self._azure_path}")

        # Load data into two dictionaries: {trial_n: abs_path, ...}
        all_trials = list(map(lambda x: join(self._azure_path, x), os.listdir(self._azure_path)))
        all_trials = list(filter(lambda x: isdir(x), all_trials))

        self._sub_trials = {int(v[-6:-4]) - 1: v for v in filter(lambda x: 'sub' in x, all_trials)}
        self._master_trials = {int(v[-9:-7]) - 1: v for v in filter(lambda x: 'master' in x, all_trials)}
        self._subject = subject

    def get_trial_by_set_nr(self, trial_nr: int):
        if trial_nr not in self._sub_trials or trial_nr not in self._master_trials:
            raise LoadingException(f"{str(self)}: Error when loading set: {trial_nr} for subject {self._azure_path}")

        sub = join(self._sub_trials[trial_nr], "positions_3d.csv")
        master = join(self._master_trials[trial_nr], "positions_3d.csv")
        return {'sub': sub, 'master': master}

    def get_nr_of_sets(self):
        return min(len(self._sub_trials), len(self._master_trials))

    def __repr__(self):
        return f"StereoAzureLoader {self._subject}"
