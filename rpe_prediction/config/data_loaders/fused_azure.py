from .base_loader import BaseSubjectLoader, LoadingException
from os.path import exists, join

import os


class FusedAzureSubjectLoader(BaseSubjectLoader):

    def __init__(self, root_path: str, subject: str):
        super().__init__()
        if not exists(root_path):
            raise LoadingException(f"Azure file not present in {root_path}")

        # Load data into dictionary: {trial_n: absolute_path, ...}
        files = list(filter(lambda x: 'azure' in x, os.listdir(root_path)))
        self._trials = {int(v.split('_')[0]): join(root_path, v) for v in files}
        self._subject = subject

    def get_trial_by_set_nr(self, trial_nr: int):
        if trial_nr not in self._trials:
            raise LoadingException(f"{str(self)}: Could not load trial: {trial_nr}")
        return self._trials[trial_nr]

    def get_nr_of_sets(self):
        return len(self._trials)

    def __repr__(self):
        return f"FusedAzureLoader {self._subject}"
