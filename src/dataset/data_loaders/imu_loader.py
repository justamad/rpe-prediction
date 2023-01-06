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

        # imu_path = join(root_path, "physilog")
        # if not exists(imu_path):
        #    raise LoadingException(f"Given directory does not exist: {imu_path}")

        json_file = join(root_path, "time_selection.json")
        if not exists(json_file):
            raise LoadingException(f"Time selection file {json_file} does not exists.")

        with open(json_file) as json_content:
            data = json.load(json_content)

        self._sets = {index: value for index, value in enumerate(data["non_truncated_selection"]["set_times"])}

        self._trials = []
        for sensor_location in sensor_locations:
            df = pd.read_csv(join(root_path, f"{sensor_location}.csv"), index_col='sensorTimestamp')
            df.index = pd.to_datetime(df.index)
            df = df.drop(columns=['Acceleration Magnitude'])
            df = df.add_prefix(sensor_location + "_")
            self._trials.append(df)

    def get_trial_by_set_nr(self, trial_nr: int):
        if trial_nr not in self._sets:
            raise LoadingException(f"Could not load trial {trial_nr} for subject {self._subject_name}")

        set_1 = self._sets[trial_nr]
        start_dt = datetime.strptime(set_1['start'], '%H:%M:%S.%f') + relativedelta(years=+70, seconds=-4)
        end_dt = datetime.strptime(set_1['end'], '%H:%M:%S.%f') + relativedelta(years=+70, seconds=4)

        # Fuse dataframes into one
        dataframes = [df.loc[(df.index > start_dt) & (df.index < end_dt)] for df in self._trials]
        max_length = min([len(df) for df in dataframes])
        df = pd.concat(
            objs=list(map(lambda x: x.iloc[:max_length].reset_index(drop=True), dataframes)),
            axis=1
        )
        df.index = dataframes[0].index[:max_length]
        return df

    def get_nr_of_sets(self):
        return len(self._trials)

    def __repr__(self):
        return f"FusedAzureLoader {self._subject_name}"
