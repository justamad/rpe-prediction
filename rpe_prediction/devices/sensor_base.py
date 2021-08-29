from rpe_prediction.processing import apply_butterworth_filter, sample_data_uniformly

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
        self._data.index += delta

    def cut_data_by_index(self, start: int = 0, end: int = -1):
        """
        Cut the data based on start and end index
        @param start: start index
        @param end: end index
        """
        self._data = self._data.iloc[start:end]

    def cut_data_by_label(self, start, end):
        """
        Cut the data by a given label using the index of data frame
        :param start: start label index
        :param end: end label index
        """
        self._data = self._data.loc[start:end]

    @property
    def sampling_frequency(self):
        return self._sampling_frequency
