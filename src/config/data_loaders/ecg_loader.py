import pyhrv.hrv

from .base_loader import BaseSubjectLoader, LoadingException
from os.path import exists, join
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pyedflib import highlevel
from tqdm import tqdm

import pandas as pd
import numpy as np
import neurokit2 as nk
import json
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


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

        acc_df = pd.read_csv(csv_file, sep=",", index_col="sensorTimestamp")
        acc_df.index = pd.to_datetime(acc_df.index)
        acc_df = acc_df.drop(columns=["Acceleration Magnitude"])

        signals, signal_headers, header = highlevel.read_edf(ecg_file)
        ecg_df = pd.DataFrame({"ECG": signals[0]})
        ecg_df.index = pd.to_datetime(ecg_df.index, unit="ms")

        self._sets = data["non_truncated_selection"]["set_times"]
        start = self._sets[0]["start"]
        end = self._sets[-1]["end"]

        start_dt = datetime.strptime(start, '%H:%M:%S.%f') + relativedelta(years=+70, seconds=-120)
        end_dt = datetime.strptime(end, '%H:%M:%S.%f') + relativedelta(years=+70, seconds=+120)

        self._acc_df = acc_df.loc[(acc_df.index > start_dt) & (acc_df.index < end_dt)]
        ecg_df = ecg_df.loc[(ecg_df.index > start_dt) & (ecg_df.index < end_dt)]

        hrv_df = self.calculate_hrv_features(
            ecg_df["ECG"].to_numpy(),
            ecg_sampling_rate=1000,
            hrv_sampling_rate=1,
            hrv_window_size=30,
        )

        index = ecg_df.index[30 * 1000::1000]
        hrv_df.index = index[:len(hrv_df)]
        self._hrv_df = hrv_df

        # for column in self._hrv_df.columns:
        #     plt.plot(self._hrv_df[column], label=column)
        #

        # plt.plot(self._hrv_df["HRV_MeanNN"], label="heart rate")
        # plt.plot(self._acc_df["ACCELERATION_X"], label="imu")
        # plt.legend()
        # plt.show()

    def get_trial_by_set_nr(self, trial_nr: int):
        if trial_nr >= len(self._sets):
            raise LoadingException(f"Couldn't load data for trial {trial_nr}.")

        set_1 = self._sets[trial_nr]
        start_dt = datetime.strptime(set_1["start"], '%H:%M:%S.%f') + relativedelta(years=+70)
        end_dt = datetime.strptime(set_1["end"], '%H:%M:%S.%f') + relativedelta(years=+70)

        result_acc_df = self._acc_df.loc[(self._acc_df.index > start_dt) & (self._acc_df.index < end_dt)]
        result_hrv_df = self._hrv_df.loc[(self._hrv_df.index > start_dt) & (self._hrv_df.index < end_dt)]

        return {
            "hrv": result_hrv_df,
            "imu": result_acc_df,
        }

    @staticmethod
    def calculate_hrv_features(
            ecg_signal: np.ndarray,
            ecg_sampling_rate: int,
            hrv_sampling_rate: int,
            hrv_window_size: int,
    ):
        ecg_clean = nk.ecg_clean(ecg_signal, sampling_rate=ecg_sampling_rate, method="neurokit")
        peaks, info = nk.ecg_peaks(
            ecg_clean,
            method="neurokit",
            sampling_rate=ecg_sampling_rate,
            correct_artifacts=True,
        )
        peaks = np.array(peaks["ECG_R_Peaks"])
        df = pd.DataFrame()

        step_size = ecg_sampling_rate // hrv_sampling_rate
        win_size = hrv_window_size * ecg_sampling_rate

        for index in tqdm(range(0, len(peaks) - win_size, step_size)):
            try:
                sub_array = peaks[index:index + win_size]
                # hrv_time = nk.hrv_time(sub_array, sampling_rate=ecg_sampling_rate, show=False)
                # hrv_freq = nk.hrv_frequency(sub_array, sampling_rate=ecg_sampling_rate, show=False, normalize=False)
                # hrv_non_linear = nk.hrv_nonlinear(sub_array, sampling_rate=ecg_sampling_rate, show=False)
                sub_df = nk.hrv(sub_array, sampling_rate=1000, show=False)
                # sub_df = pd.concat([hrv_time, hrv_freq, hrv_non_linear], axis=1)
                df = pd.concat([df, sub_df], ignore_index=False)

            except Exception as e:
                print(f"Error for window: {index} with message {e}")

        return df

    def get_nr_of_sets(self):
        return len(self._sets)

    def __repr__(self):
        return f"ECG Loader {self._subject_name}"
