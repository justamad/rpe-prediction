from .base_loader import BaseSubjectLoader, LoadingException
from os.path import exists, join
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pyedflib import highlevel

import pandas as pd
import json


class ECGLoader(BaseSubjectLoader):

    def __init__(self, root_path: str, subject: str):
        super().__init__()
        if not exists(root_path):
            raise LoadingException(f"Azure file not present in {root_path}")

        ecg_file = join(root_path, "ecg.edf")
        csv_file = join(root_path, "FAROS.csv")
        json_file = join(root_path, "time_selection.json")

        if not exists(ecg_file):
            raise LoadingException(f"ECG file {ecg_file} not found.")

        if not exists(csv_file):
            raise LoadingException(f"Faros acceleration file {csv_file} not found.")

        if not exists(json_file):
            raise LoadingException(f"JSON file with temporal definitions {json_file} not found.")

        with open(json_file) as json_content:
            data = json.load(json_content)

        self._sets = data["non_truncated_selection"]["set_times"]
        self._subject = subject

        signals, signal_headers, header = highlevel.read_edf(ecg_file)
        self._df_edf = pd.DataFrame(data=signals[0])
        self._df_edf.index = pd.to_datetime(self._df_edf.index, unit="ms")

        self._df_acc = pd.read_csv(csv_file, sep=',', index_col="sensorTimestamp")
        self._df_acc.index = pd.to_datetime(self._df_acc.index)

    def get_trial_by_set_nr(self, trial_nr: int):
        if trial_nr >= len(self._sets):
            raise LoadingException(f"Couldn't load data for trial {trial_nr}.")

        set_1 = self._sets[trial_nr]
        start_dt = datetime.strptime(set_1['start'], '%H:%M:%S.%f') + relativedelta(years=+70)
        end_dt = datetime.strptime(set_1['end'], '%H:%M:%S.%f') + relativedelta(years=+70)

        result_acc_df = self._df_acc.loc[(self._df_acc.index > start_dt) & (self._df_acc.index < end_dt)]
        result_ecg_df = self._df_edf.loc[(self._df_edf.index > start_dt) & (self._df_edf.index < end_dt)]
        return result_ecg_df, result_acc_df

    def get_nr_of_sets(self):
        return len(self._sets)

    def __repr__(self):
        return f"ECG Loader {self._subject}"
