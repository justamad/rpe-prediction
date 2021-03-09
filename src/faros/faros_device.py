from src.processing import apply_butterworth_filter, normalize_signal, find_peaks, sample_data_uniformly
from pyedflib import highlevel
from biosppy.signals import ecg
from os.path import join

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def read_acceleration_freq_from_file(signal_headers):
    sample_rates = list(map(lambda h: h["sample_rate"], filter(lambda h: "Acc" in h["label"], signal_headers)))
    if any([sr != sample_rates[0] for sr in sample_rates]):
        raise UserWarning(f"Not all Faros accelerometer sampling rates are the same: {sample_rates}")

    return sample_rates[0]


def read_ecg_freq_from_file(signal_headers):
    sampling_freq = [signal['sample_rate'] for signal in signal_headers if signal['label'] == 'ECG']
    assert len(sampling_freq) == 1, f"No or too many ECG signals: {sampling_freq}"
    return sampling_freq[0]


def read_acceleration_data_from_file(signals, signal_headers):
    headers = ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']
    data = []
    for column in headers:
        idx = [header['label'] for header in signal_headers].index(column)
        data.append(signals[idx])

    data_body = np.stack(data, axis=1)
    return pd.DataFrame(data_body, columns=headers)


def read_ecg_signal(signals, signal_headers):
    ecg_signal = [signal['label'] for signal in signal_headers].index("ECG")
    return signals[ecg_signal]


class Faros(object):

    def __init__(self, folder_path, ekg_path=None, sampling_rate_imu=100, sampling_rate_hr=500):
        if isinstance(folder_path, pd.DataFrame):
            self.acc_data = folder_path
            self.ecg_data = folder_path
        elif isinstance(folder_path, str):
            file_name = join(folder_path, "record.edf")
            if not os.path.exists(file_name):
                raise Exception(f"Folder {file_name} does not exist.")

            signals, signal_headers, _ = highlevel.read_edf(file_name)
            sampling_rate_imu = read_acceleration_freq_from_file(signal_headers)
            self.acc_data = read_acceleration_data_from_file(signals, signal_headers)
            self._sampling_frequency = sampling_rate_imu
            self._sampling_frequency_ecg = sampling_rate_hr

            self.ecg_data = read_ecg_signal(signals, signal_headers)
            hr_x, hr = self.get_heart_rate_signal()
            hr = pd.DataFrame({'hr': hr})

            hr, hr_x = sample_data_uniformly(hr, hr_x, 100, mode="linear")
            acc = self.acc_data['Accelerometer_X'].to_numpy()
            plt.plot(np.arange(len(acc)) / 100, normalize_signal(acc))
            plt.plot(hr_x, normalize_signal(hr))
            plt.show()
            self.file_name = file_name
        else:
            raise Exception(f"Unknown argument {folder_path} for Faros class.")

        # Peak finding parameters
        self.height = 1.2
        self.prominence = 1.0
        self.distance = None

    def get_acceleration_data(self):
        data_np = self.acc_data.to_numpy()
        return data_np

    def get_synchronization_signal(self) -> np.ndarray:
        data = self.acc_data['Accelerometer_X'].to_numpy()
        return data

    def get_timestamps(self) -> np.ndarray:
        return np.arange(len(self.acc_data)) / self.sampling_frequency

    def get_synchronization_data(self):
        clock = self.get_timestamps()
        raw_signal = apply_butterworth_filter(self.get_synchronization_signal())
        raw_signal = normalize_signal(raw_signal)
        processed_signal = -raw_signal  # Inversion of coordinate system
        peaks = find_peaks(-processed_signal, height=self.height, prominence=self.prominence)
        return clock, raw_signal, processed_signal, peaks

    def get_heart_rate_signal(self):
        heart_rate = ecg.ecg(self.ecg_data, sampling_rate=self._sampling_frequency_ecg, show=False)
        return heart_rate['heart_rate_ts'], heart_rate['heart_rate']

    def cut_trial(self, start_idx, end_idx):
        data = self.acc_data.iloc[start_idx:end_idx]
        return data

    @property
    def sampling_frequency(self):
        return self._sampling_frequency

    def __repr__(self):
        return "Faros"
