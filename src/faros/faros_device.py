from typing import List, Tuple

from src.processing import apply_butterworth_filter, normalize_signal, find_peaks
from pyedflib import highlevel

import pandas as pd
import numpy as np
import os


def read_ecg_freq_from_file(signal_headers):
    sampling_freq = [signal['sample_rate'] for signal in signal_headers if signal['label'] == 'ECG']
    assert len(sampling_freq) == 1, f"No or too many ECG signals: {sampling_freq}"
    return sampling_freq[0]


def read_acceleration_freq_from_file(signal_headers):
    sample_rates = list(map(lambda h: h["sample_rate"], filter(lambda h: "Acc" in h["label"], signal_headers)))
    if any([sr != sample_rates[0] for sr in sample_rates]):
        raise UserWarning(f"Not all Faros accelerometer sampling rates are the same: {sample_rates}")
    return sample_rates[0]


def read_acceleration_data_from_file(signals, signal_headers):
    headers = ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']
    data = []
    for column in headers:
        idx = [header['label'] for header in signal_headers].index(column)
        data.append(signals[idx])

    data_body = np.stack(data, axis=1)
    return pd.DataFrame(data_body, columns=headers)


def get_signal_from_datasource(signals, signal_headers, label):
    idx = [header['label'] for header in signal_headers].index(label)
    return signals[idx]


class Faros(object):

    def __init__(self, folder, start=0, end=-1):
        if not os.path.exists(folder):
            raise Exception(f"Folder {folder} does not exist.")

        file_name = os.path.join(folder, "record.edf")
        signals, signal_headers, header = highlevel.read_edf(file_name)
        self.acc_data = read_acceleration_data_from_file(signals, signal_headers).iloc[start:end]
        self._sampling_frequency = read_acceleration_freq_from_file(signal_headers)
        self.height = 1.2
        self.prominence = 1.0
        self.distance = None

    def get_acceleration_data(self) -> List[Tuple[np.ndarray, str]]:
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

    @property
    def sampling_frequency(self):
        return self._sampling_frequency

    def __repr__(self):
        return "Faros"
