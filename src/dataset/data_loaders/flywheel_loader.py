from .base_loader import BaseSubjectLoader, LoadingException
from os.path import join, isfile

import json
import pandas as pd
import logging


class FlyWheelSubjectLoader(BaseSubjectLoader):

    def __init__(self, root_path: str, subject_name: str):
        super().__init__(subject_name)

        flywheel_file = join(root_path, "kmeter.json")
        with open(flywheel_file) as f:
            self._content = json.load(f)

        if len(self._content) != 12:
            logging.warning("Flywheel - maybe not all trials are captured.")

    def get_nr_of_sets(self):
        return len(self._content)

    def get_trial_by_set_nr(self, trial_nr: int):
        if trial_nr > len(self._content):
            raise LoadingException(f"{str(self)}: Could not load trial {trial_nr}")

        set_data = self._content[trial_nr]

        set_df = pd.concat([pd.Series(rep) for rep in set_data["training_rep"]], axis=1, ignore_index=True).T
        set_df.drop(["entry_time", "id", "is_old_data", "set_id", "status"], axis=1, inplace=True)

        set_df = set_df.astype(float)
        return set_df

    def __repr__(self):
        return f"RPELoader {self._subject_name}"
