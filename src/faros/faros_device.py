from typing import List, Tuple

from src.processing import apply_butterworth_filter, normalize_signal, find_peaks, utils
from pyedflib import highlevel

import pandas as pd
import numpy as np
import os


class Faros(object):

    def __init__(self, folder, start=0, end=-1):
        if not os.path.exists(folder):
            raise Exception(f"Folder {folder} does not exist.")

        file_name = os.path.join(folder, "record.edf")
        self.acc_data = self.read_acceleration_data_from_file(file_name).iloc[start:end]
        self.file_name = file_name
        self._sampling_frequency = self.read_acceleration_freq_from_file(file_name)
        self.height = 1.2
        self.prominence = 1.0
        self.distance = None

    def get_acceleration_data(self, add_magnitude=False) -> List[Tuple[np.ndarray, str]]:
        data_np = self.acc_data.to_numpy()
        if add_magnitude:
            return [(utils.add_magnitude(data_np), "Faros")]
        else:
            return [(data_np, "Faros")]

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

    @staticmethod
    def read_acceleration_freq_from_file(file_name):
        _, signal_headers, _ = highlevel.read_edf(file_name)
        sample_rates = list(map(lambda h: h["sample_rate"], filter(lambda h: "Acc" in h["label"], signal_headers)))
        if any([sr != sample_rates[0] for sr in sample_rates]):
            raise UserWarning(f"Not all Faros accelerometer sampling rates are the same: {sample_rates}")
        return sample_rates[0]

    @staticmethod
    def read_acceleration_data_from_file(file_name):
        signals, signal_headers, header = highlevel.read_edf(file_name)

        headers = ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']
        data = []
        for column in headers:
            idx = [header['label'] for header in signal_headers].index(column)
            data.append(signals[idx])

        data_body = np.stack(data, axis=1)
        return pd.DataFrame(data_body, columns=headers)

    @staticmethod
    def get_signal_from_datasource(file_name, label):
        signals, signal_headers, header = highlevel.read_edf(file_name)
        idx = [header['label'] for header in signal_headers].index(label)
        return signals[idx]

    @property
    def sampling_frequency(self):
        return self._sampling_frequency

    def __repr__(self):
        return "Faros"
