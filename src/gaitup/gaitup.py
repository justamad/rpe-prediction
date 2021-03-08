from typing import Tuple, Any
from src.processing import normalize_signal, find_peaks, sample_data_uniformly, apply_butterworth_filter_dataframe
from os.path import join

import pandas as pd
import numpy as np
import os


class GaitUp(object):

    def __init__(self, data, sampling_frequency=100):
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, str):
            if not os.path.exists(data):
                raise Exception(f"Folder {data} does not exist.")

            data_frames = []
            for file_name in os.listdir(data):
                df = pd.read_csv(join(data, file_name), delimiter=',')
                df = df[[c for c in df.columns if "Time" not in c and "Event" not in c]]
                df = df.add_prefix(file_name.replace('.csv', '') + "_")
                data_frames.append(df)

            data = pd.concat(data_frames, join='outer', axis=1)
            data = sample_data_uniformly(data, timestamps=np.arange(len(data)) / 128, sampling_rate=sampling_frequency)
            self.data = apply_butterworth_filter_dataframe(data, sampling_frequency=sampling_frequency)

        else:
            raise Exception(f"Unknown argument to create Gaitup object: {data}")

        self._sampling_frequency = sampling_frequency
        self.height = 0.5
        self.prominence = 2
        self.distance = 140

    def cut_data(self, start_idx, end_idx):
        data = self.data.iloc[start_idx:end_idx]
        return GaitUp(data, self.sampling_frequency)

    def get_synchronization_signal(self):
        return self.data['ST327_Accel Y'].to_numpy()

    def get_timestamps(self) -> np.ndarray:
        return np.arange(len(self.data)) / self.sampling_frequency

    def get_synchronization_data(self) -> Tuple[np.ndarray, Any, Any, np.ndarray]:
        clock = self.get_timestamps()
        raw_signal = self.get_synchronization_signal()
        raw_signal = normalize_signal(raw_signal)
        processed_signal = -raw_signal
        peaks = find_peaks(-processed_signal, height=self.height, prominence=self.prominence, distance=self.distance)
        return clock, raw_signal, processed_signal, peaks

    @property
    def sampling_frequency(self):
        return self._sampling_frequency

    def __repr__(self):
        return "GaitUp"
