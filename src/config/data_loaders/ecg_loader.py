from .base_loader import BaseSubjectLoader, LoadingException
from os.path import exists, join
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pyedflib import highlevel

import pandas as pd
import neurokit2 as nk
import json


class ECGSubjectLoader(BaseSubjectLoader):

    def __init__(self, root_path: str, subject_name: str):
        super().__init__(subject_name)
        if not exists(root_path):
            raise LoadingException(f"Given folder {root_path} does not exist.")

        ecg_file = join(root_path, f"ecg-{subject_name}.edf")
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

        signals, signal_headers, header = highlevel.read_edf(ecg_file)
        self._df_edf = pd.DataFrame({'ecg': signals[0]})
        self._df_edf.index = pd.to_datetime(self._df_edf.index, unit="ms")

        df_acc = pd.read_csv(csv_file, sep=',', index_col="sensorTimestamp")
        df_acc.index = pd.to_datetime(df_acc.index)
        self._df_acc = df_acc.drop(columns=['Acceleration Magnitude'])

        ecg_cleaned = nk.ecg_clean(self._df_edf['ecg'], sampling_rate=1000, method='elgendi2010')
        _, r_peaks = nk.ecg_peaks(ecg_cleaned, method='elgendi2010', sampling_rate=1000, correct_artifacts=True)
        peaks = r_peaks['ECG_R_Peaks']

    def get_trial_by_set_nr(self, trial_nr: int):
        if trial_nr >= len(self._sets):
            raise LoadingException(f"Couldn't load data for trial {trial_nr}.")

        set_1 = self._sets[trial_nr]
        start_dt = datetime.strptime(set_1['start'], '%H:%M:%S.%f') + relativedelta(years=+70, seconds=-40)
        end_dt = datetime.strptime(set_1['end'], '%H:%M:%S.%f') + relativedelta(years=+70, seconds=4)

        result_acc_df = self._df_acc.loc[(self._df_acc.index > start_dt) & (self._df_acc.index < end_dt)]
        result_ecg_df = self._df_edf.loc[(self._df_edf.index > start_dt) & (self._df_edf.index < end_dt)]

        return {
            "ecg": result_ecg_df,
            "imu": result_acc_df,
        }

    def get_nr_of_sets(self):
        return len(self._sets)

    def __repr__(self):
        return f"ECG Loader {self._subject_name}"
