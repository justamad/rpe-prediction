from src.processing import normalize_signal, apply_butterworth_filter, find_closest_timestamp
from ..sensor_base import SensorBase
from os.path import join

import pandas as pd
import os


def read_directory_with_csv_file(folder_name):
    data_frames = []
    for file_name in os.listdir(folder_name):
        df = pd.read_csv(join(folder_name, file_name), delimiter=',')
        if len(data_frames) == 0:
            df = df[[c for c in df.columns if "Event" not in c]]
        else:
            df = df[[c for c in df.columns if "Time" not in c and "Event" not in c]]

        prefix = f"{file_name.replace('.csv', '')}_"
        df.columns = ["{}{}".format('' if c == 'Time' else prefix, c) for c in df.columns]
        data_frames.append(df)

    data = pd.concat(data_frames, join='outer', axis=1)
    data.rename(columns={"Time": "timestamp"})
    return data


class GaitUp(SensorBase):

    def __init__(self, data, sampling_frequency=128):
        if isinstance(data, pd.DataFrame):
            data = data
        elif isinstance(data, str):
            if not os.path.exists(data):
                raise Exception(f"Folder {data} does not exist.")

            data = read_directory_with_csv_file(data)
        else:
            raise Exception(f"Unknown argument to create Gaitup object: {data}")

        super().__init__(data, sampling_frequency)

    def cut_data_based_on_time(self, start_time, end_time):
        start_idx = find_closest_timestamp(self.timestamps, start_time)
        end_idx = find_closest_timestamp(self.timestamps, end_time)
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
        return self.timestamps, raw_signal, processed_signal

    def shift_clock(self, delta):
        """
        Shift the clock based on a given time delta
        @param delta: the time offset given in seconds
        """
        self.data[['Time']] += delta

    @property
    def timestamps(self):
        return self.data['Time'].to_numpy()

    def __repr__(self):
        """
        Returns a string representation for gait up device
        @return: string representation
        """
        return "GaitUp"
