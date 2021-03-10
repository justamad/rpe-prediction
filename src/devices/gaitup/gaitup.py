from src.processing import normalize_signal, apply_butterworth_filter, find_closest_timestamp
from os.path import join

import pandas as pd
import numpy as np
import os


class GaitUp(object):

    def __init__(self, data, sampling_frequency=128):
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

            self.data = pd.concat(data_frames, join='outer', axis=1)

        else:
            raise Exception(f"Unknown argument to create Gaitup object: {data}")

        self._sampling_frequency = sampling_frequency
        self._timestamps = np.arange(len(self.data)) / self.sampling_frequency
        self.height = 0.5
        self.prominence = 2
        self.distance = 140

    def cut_data_based_on_time(self, start_time, end_time):
        start_idx = find_closest_timestamp(self._timestamps, start_time)
        end_idx = find_closest_timestamp(self._timestamps, end_time)
        return self.cut_data_based_on_index(start_idx, end_idx)

    def cut_data_based_on_index(self, start_idx, end_idx):
        data = self.data.iloc[start_idx:end_idx]
        return GaitUp(data, self.sampling_frequency)

    def get_synchronization_signal(self):
        return self.data['ST327_Accel Y'].to_numpy()

    def get_synchronization_data(self):
        raw_signal = apply_butterworth_filter(self.get_synchronization_signal())
        raw_signal = normalize_signal(raw_signal)
        processed_signal = -raw_signal
        return self._timestamps, raw_signal, processed_signal

    def shift_clock(self, delta):
        """
        Shift the clock based on a given delta
        @param delta:
        """
        self._timestamps += delta

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def sampling_frequency(self):
        return self._sampling_frequency

    def __repr__(self):
        return "GaitUp"
