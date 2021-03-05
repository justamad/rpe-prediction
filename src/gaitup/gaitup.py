from typing import Tuple, Any, List
from src.processing import apply_butterworth_filter, normalize_signal, find_peaks, utils

import pandas as pd
import numpy as np
import os


class GaitUp(object):

    def __init__(self, folder, synchronization_file_name: str = "ST327.csv"):
        if not os.path.exists(folder):
            raise Exception(f"Folder {folder} does not exist.")

        self.files = {}
        for csv_file in os.listdir(folder):
            data = pd.read_csv(os.path.join(folder, csv_file), delimiter=',')
            print(len(data))
            self.files[csv_file] = data

        self._sampling_frequency = 128
        self.height = 0.5
        self.prominence = 2
        self.distance = 140
        self.synchronization_file_name = synchronization_file_name

    def get_synchronization_signal(self):
        data = self.files[self.synchronization_file_name]['Accel Y'].to_numpy()
        return data

    def get_timestamps(self) -> np.ndarray:
        return np.arange(len(self.files[self.synchronization_file_name])) / self.sampling_frequency

    def get_acceleration_data(self, add_magnitude=False) -> List[Tuple[np.ndarray, str]]:
        ret_data = []
        for file_name, data in self.files.items():
            if "HA" in file_name:
                continue

            data = data[['Accel X', 'Accel Y', 'Accel Z']].to_numpy()
            if add_magnitude:
                ret_data.append((utils.add_magnitude(data), file_name))
            else:
                ret_data.append((data, file_name))

        return ret_data

    def get_rotation_data(self, add_magnitude=False) -> List[Tuple[np.ndarray, str]]:
        ret_data = []
        for file_name, data in self.files.items():
            if "HA" in file_name:
                continue

            data = data[['Gyro X', 'Gyro Y', 'Gyro Z']].to_numpy()
            if add_magnitude:
                ret_data.append((utils.add_magnitude(data), file_name))
            else:
                ret_data.append((data, file_name))

        return ret_data

    def get_tagging_data(self) -> List[Tuple[np.ndarray, str]]:
        ret_data = []
        for file_name, data in self.files.items():
            if "HA" not in file_name:
                continue

            data = data[['Event']].to_numpy()
            ret_data.append((data, file_name))

        return ret_data

    def get_synchronization_data(self) -> Tuple[np.ndarray, Any, Any, np.ndarray]:
        clock = self.get_timestamps()
        raw_signal = apply_butterworth_filter(self.get_synchronization_signal())
        raw_signal = normalize_signal(raw_signal)
        processed_signal = -raw_signal
        peaks = find_peaks(-processed_signal, height=self.height, prominence=self.prominence, distance=self.distance)
        return clock, raw_signal, processed_signal, peaks

    @property
    def sampling_frequency(self):
        return self._sampling_frequency

    def __repr__(self):
        return "GaitUp"
