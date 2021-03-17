from src.devices.processing import apply_butterworth_filter_dataframe

import pandas as pd


class SensorBase(object):

    def __init__(self, data: pd.DataFrame, sampling_frequency: int):
        self._data = data
        self._sampling_frequency = sampling_frequency

    def filter_data(self, order: int = 4, fc: int = 6):
        """
        Apply a butterworth filter to the data frame
        @param order: the order of the Butterworth filter
        @param fc: the cut-off frequency used in Butterworth filter
        """
        data = self._data[[c for c in self._data.columns if c != 'timestamp']].copy()
        data = apply_butterworth_filter_dataframe(data, self._sampling_frequency, order, fc)
        self._data.update(data)
        self._data = self._data.iloc[10:-10]  # Cut off the edge effects of filtering

    def save_data_as_csv(self, file_name: str):
        """
        Save the current data frame as csv file
        @param file_name: the desired file name for the csv file
        """
        self._data.to_csv(file_name, sep=";", index=False)

    @property
    def timestamps(self):
        return self._data['timestamp'].to_numpy()

    @property
    def sampling_frequency(self):
        return self._sampling_frequency
