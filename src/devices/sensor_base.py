from src.processing import apply_butterworth_filter_dataframe

import pandas as pd


class SensorBase(object):

    def __init__(self, data: pd.DataFrame, sampling_frequency: int):
        self.data = data
        self._sampling_frequency = sampling_frequency

    def filter_data(self):
        data_body = self.data[[c for c in self.data.columns if c != "timestamp"]]
        filtered = apply_butterworth_filter_dataframe(data_body, self.sampling_frequency)
        self.data.update(filtered)

    @property
    def timestamps(self):
        return self.data['timestamp'].to_numpy()

    @property
    def sampling_frequency(self):
        return self._sampling_frequency
