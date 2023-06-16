from .base_loader import BaseSubjectLoader, LoadingException
from os.path import join, isfile

import json


class RPESubjectLoader(BaseSubjectLoader):

    def __init__(self, root_path: str, subject_name: str):
        super().__init__(subject_name)
        json_file = join(root_path, "rpe_ratings.json")
        if not isfile(json_file):
            raise LoadingException(f"RPE file {json_file} does not exists!")

        with open(json_file) as f:
            rpe_values = json.load(f)

        self._trials = {k: v for k, v in enumerate(rpe_values['rpe_ratings'])}

    def get_nr_of_sets(self):
        return len(self._trials)

    def get_trial_by_set_nr(self, trial_nr: int):
        if trial_nr not in self._trials:
            raise LoadingException(f"{str(self)}: Could not load trial {trial_nr}")
        return self._trials[trial_nr]

    def __repr__(self):
        return f"RPELoader {self._subject_name}"
