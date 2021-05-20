from rpe_prediction.processing import apply_butterworth_filter, normalize_signal, sample_data_uniformly, find_closest_timestamp
from .sensor_base import SensorBase
from pyedflib import highlevel
from biosppy.signals import ecg
from os.path import join

import pandas as pd
import numpy as np
import os


class Faros(SensorBase):

    def __init__(self, folder, start=0, end=-1):
        """
        Constructor for Faros Device
        @param folder: folder where faros device resides in
        @param start: start index
        @param end: end index
        """
        file_name = join(folder, "record.EDF")
        if not os.path.exists(file_name):
            raise Exception(f"Faros file {file_name} does not exist.")

        signals, signal_headers, header = highlevel.read_edf(file_name)

        # Set acceleration properties
        acceleration_data, sampling_frequency = self.read_acceleration_data(signal_headers, signals)
        acceleration_data = acceleration_data.iloc[start:end]
        super().__init__(acceleration_data, sampling_frequency)

        # Set and calculate heart rate properties
        sampling_frequency_ecg = self._read_sampling_rate_for_label(signal_headers, "ECG")
        ecg_factor = sampling_frequency_ecg // sampling_frequency
        # ecg_data = self._read_signal_for_label(signals, signal_headers, "ECG")[start * ecg_factor:end * ecg_factor]
        # self.timestamps_hr, self.hr_data = self._calculate_heart_rate_signal(ecg_data, sampling_frequency_ecg, 100)

        # Read HRV Properties
        # self._sampling_frequency_hrv = self._read_sampling_rate_for_label(signal_headers, "HRV")
        # self.hrv_data = self._read_signal_for_label(signals, signal_headers, "HRV")
        # self.timestamps_hrv = np.arange(len(self.hrv_data)) / self._sampling_frequency_hrv

    def read_acceleration_data(self, signal_headers, signals):
        sampling_frequency = self._read_sampling_rate_for_label(signal_headers, "Acc")

        data = []
        for label in ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']:
            data.append(self._read_signal_for_label(signals, signal_headers, label))

        timestamps = np.arange(len(data[0])) / sampling_frequency
        data.insert(0, timestamps)
        data = np.stack(data, axis=1)
        return pd.DataFrame(data, columns=["timestamp", "acc (x)", "acc (y)", "acc (z)"]), sampling_frequency

    def cut_data_based_on_time(self, start_time, end_time):
        pass
        # start_idx = find_closest_timestamp(self.timestamps_hr, start_time)
        # end_idx = find_closest_timestamp(self.timestamps_hr, end_time)
        # self.hr_data = self.hr_data[start_idx:end_idx]
        # self.timestamps_hr = self.timestamps_hr[start_idx:end_idx]

    def shift_clock(self, delta):
        """
        Shift the clock based on a given time delta
        @param delta: the time offset given in seconds
        """
        self._data.loc[:, 'timestamp'] += delta

    def get_synchronization_signal(self) -> np.ndarray:
        """
        Return a 1-D signal for synchronization using the most representative axis
        @return: 1d numpy array with sensor data
        """
        return self._data['acc (x)'].to_numpy()

    def get_synchronization_data(self):
        raw_signal = apply_butterworth_filter(self.get_synchronization_signal())
        raw_signal = normalize_signal(raw_signal)
        processed_signal = -raw_signal  # Inversion of coordinate system
        return self.timestamps, raw_signal, processed_signal

    def __repr__(self):
        """
        String representation of Faros class
        @return: string with sensor name
        """
        return "Faros"

    @staticmethod
    def _calculate_heart_rate_signal(ecg_data, sampling_frequency_ecg, sampling_frequency_hr, show=False):
        heart_rate = ecg.ecg(ecg_data, sampling_rate=sampling_frequency_ecg, show=show)
        hr_x, hr = heart_rate['heart_rate_ts'], pd.DataFrame({'hr': heart_rate['heart_rate']}),
        data, timestamps = sample_data_uniformly(hr, hr_x, sampling_rate=sampling_frequency_hr, mode="quadratic")
        return timestamps, data

    @staticmethod
    def _read_sampling_rate_for_label(signal_headers, label="Acc"):
        sample_rates = list(map(lambda h: h["sample_rate"], filter(lambda h: label in h["label"], signal_headers)))
        if any([sr != sample_rates[0] for sr in sample_rates]):
            raise UserWarning(f"Not all Faros accelerometer sampling rates are the same: {sample_rates}")
        return sample_rates[0]

    @staticmethod
    def _read_signal_for_label(signals, signal_headers, label):
        """
        Reads the time series data from given signals and headers for a given label
        @param signals: all signals available by the faros sensor
        @param signal_headers: header information for all faros sensor modalities
        @param label: label of the desired signal
        @return: 1D-array of desired signal
        """
        signal_idx = [signal['label'] for signal in signal_headers].index(label)
        return signals[signal_idx]
