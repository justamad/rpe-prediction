from src.devices.processing import apply_butterworth_filter_dataframe, sample_data_uniformly

import pandas as pd


class SensorBase(object):

    def __init__(self, data: pd.DataFrame, sampling_frequency: int):
        """
        Constructor for the Sensor Base class.
        Each sensor should have a basis data frame with timestamps in the first column
        :param data: the current sensor data in a data frame
        :param sampling_frequency: the current sampling frequency of the sensor
        """
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

    def resample_data(self, sampling_frequency):
        self._data = sample_data_uniformly(self._data, sampling_rate=sampling_frequency)
        self._sampling_frequency = sampling_frequency

    def save_data_as_csv(self, file_name: str):
        """
        Save the current data frame as csv file
        @param file_name: the desired file name for the csv file
        """
        self._data.to_csv(file_name, sep=";", index=False)

    def shift_clock(self, delta):
        """
        Shift the clock based on a given time delta
        @param delta: the time offset given in seconds
        """
        self._data.loc[:, 'timestamp'] += delta

    def cut_data_by_index(self, start: int = 0, end: int = -1):
        """
        Cut the data based on start and end index
        @param start: start index
        @param end: end index
        """
        self._data = self._data.iloc[start:end]

    @property
    def timestamps(self):
        return self._data['timestamp'].to_numpy()

    @property
    def sampling_frequency(self):
        return self._sampling_frequency
