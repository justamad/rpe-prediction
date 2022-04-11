from .base_loader import BaseSubjectLoader, LoadingException
from os.path import exists, join

import pandas as pd
import neurokit2 as nk
import os


class DataFrameLoader(BaseSubjectLoader):

    def __init__(
            self,
            root_path: str,
            subject_name: str,
            sensor_name: str,
    ):
        super().__init__(subject_name)
        if not exists(root_path):
            raise LoadingException(f"Given path does not exist: {root_path}.")

        # Load data into dictionary: {trial_n: absolute_path, ...}
        files = list(filter(lambda x: sensor_name in x, os.listdir(root_path)))
        self._trials = {int(v.split('_')[0]): join(root_path, v) for v in files}

    def get_trial_by_set_nr(self, trial_nr: int):
        if trial_nr not in self._trials:
            raise LoadingException(f"{str(self)}: Could not load trial: {trial_nr}")

        df = pd.read_csv(self._trials[trial_nr], sep=";", index_col="timestamp", parse_dates=True)
        # df.index = pd.to_datetime(df.index, unit="s")
        return df

    def get_nr_of_sets(self):
        return len(self._trials)

    def __repr__(self):
        return f"DataFrameLoader {self._subject_name}"


class AzureDataFrameLoader(DataFrameLoader):

    def __init__(self, root_path: str, subject_name: str):
        super().__init__(root_path, subject_name, "azure")

    def __repr__(self):
        return f"AzureDFLoader {self._subject_name}"


class IMUDataFrameLoader(DataFrameLoader):

    def __init__(self, root_path: str, subject_name: str):
        super().__init__(root_path, subject_name, "imu")

    def get_trial_by_set_nr(self, trial_nr: int):
        if trial_nr not in self._trials:
            raise LoadingException(f"{str(self)}: Could not load trial: {trial_nr}")

        df = pd.read_csv(self._trials[trial_nr], sep=";", index_col="sensorTimestamp")
        df.index = pd.to_datetime(df.index)
        return df

    def __repr__(self):
        return f"IMUDFLoader {self._subject_name}"


class ECGDataFrameLoader(DataFrameLoader):

    def __init__(self, root_path: str, subject_name: str):
        super().__init__(root_path, subject_name, "ecg")

    def get_trial_by_set_nr(self, trial_nr: int):
        if trial_nr not in self._trials:
            raise LoadingException(f"{str(self)}: Could not load trial: {trial_nr}")

        df = pd.read_csv(self._trials[trial_nr], sep=";", index_col="sensorTimestamp")
        df.index = pd.to_datetime(df.index)

        # ecg_cleaned = nk.ecg_clean(self._df_edf['ecg'], sampling_rate=1000, method="neurokit2")
        # _, r_peaks = nk.ecg_peaks(ecg_cleaned, method="neurokit2", sampling_rate=1000, correct_artifacts=True)
        # peaks = r_peaks["ECG_R_Peaks"]

        return df

    def __repr__(self):
        return f"ECGDFLoader {self._subject_name}"
