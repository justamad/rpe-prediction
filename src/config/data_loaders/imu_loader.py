from .base_loader import BaseSubjectLoader, LoadingException
from os.path import exists, join
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import json

sensor_locations = ["CHEST", "LUMBAR SPINE", "THIGH, LEFT", "THIGH, RIGHT", "TIBIA, LEFT", "TIBIA, RIGHT"]


class IMUSubjectLoader(BaseSubjectLoader):

    def __init__(self, root_path: str, subject_name: str):
        super().__init__(subject_name)
        if not exists(root_path):
            raise LoadingException(f"Given directory does not exist: {root_path}")

        json_file = join(root_path, "time_selection.json")

        with open(json_file) as json_content:
            data = json.load(json_content)

        self._sets = data["non_truncated_selection"]["set_times"]

        self._trials = {}
        for sensor_location in sensor_locations:
            df = pd.read_csv(join(root_path, f"{sensor_location}.csv"), index_col='sensorTimestamp')
            df.index = pd.to_datetime(df.index)
            df = df.drop(columns=['Acceleration Magnitude'])
            self._trials[sensor_location] = df

    def get_trial_by_set_nr(self, trial_nr: int):
        set_1 = self._sets[trial_nr]
        start_dt = datetime.strptime(set_1['start'], '%H:%M:%S.%f') + relativedelta(years=+70, seconds=-4)
        end_dt = datetime.strptime(set_1['end'], '%H:%M:%S.%f') + relativedelta(years=+70, seconds=4)

        ret_dict = {}
        for sensor_location, df in self._trials.items():
            selection = df.loc[(df.index > start_dt) & (df.index < end_dt)]
            ret_dict[sensor_location] = selection

        return ret_dict

    def get_nr_of_sets(self):
        return len(self._trials)

    def __repr__(self):
        return f"FusedAzureLoader {self._subject_name}"
