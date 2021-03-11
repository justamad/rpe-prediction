from src.processing import apply_butterworth_filter, normalize_signal, sample_data_uniformly, find_closest_timestamp
from pyedflib import highlevel
from biosppy.signals import ecg
from ..sensor_base import SensorBase
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
        self._sampling_frequency_imu = self.read_acceleration_freq_from_file(signal_headers)
        self.acc_data = self.read_acceleration_data_from_file(signals, signal_headers).iloc[start:end]
        self._timestamps_imu = np.arange(len(self.acc_data)) / self._sampling_frequency_imu

        # Set and calculate heart rate properties
        self._sampling_frequency_ecg = self.read_ecg_freq_from_file(signal_headers)
        ecg_factor = self._sampling_frequency_ecg // self._sampling_frequency_imu
        ecg_data = self.read_ecg_signal(signals, signal_headers)[start*ecg_factor:end*ecg_factor]
        self.timestamps_hr, self.hr_data = self.calculate_heart_rate_signal(ecg_data, self._sampling_frequency_ecg, 100)
        super().__init__(ecg_data, self._sampling_frequency_imu)

    def get_acceleration_data(self):
        return self.acc_data.to_numpy()

    def cut_data_based_on_time(self, start_time, end_time):
        start_idx = find_closest_timestamp(self.timestamps_hr, start_time)
        end_idx = find_closest_timestamp(self.timestamps_hr, end_time)
        self.hr_data = self.hr_data[start_idx:end_idx]
        self.timestamps_hr = self.timestamps_hr[start_idx:end_idx]

    def add_shift(self, shift):
        """
        Add a shift to both clocks
        @param shift: shift in seconds to add
        """
        self._timestamps_imu += shift
        self.timestamps_hr += shift

    def get_synchronization_signal(self) -> np.ndarray:
        return self.acc_data['Accelerometer_X'].to_numpy()

    def get_synchronization_data(self):
        raw_signal = apply_butterworth_filter(self.get_synchronization_signal())
        raw_signal = normalize_signal(raw_signal)
        processed_signal = -raw_signal  # Inversion of coordinate system
        return self._timestamps_imu, raw_signal, processed_signal

    def __repr__(self):
        """
        String representation of Faros class
        @return: string with sensor name
        """
        return "Faros"

    @staticmethod
    def calculate_heart_rate_signal(ecg_data, sampling_frequency_ecg, sampling_frequency_hr):
        heart_rate = ecg.ecg(ecg_data, sampling_rate=sampling_frequency_ecg, show=False)
        hr_x, hr = heart_rate['heart_rate_ts'], pd.DataFrame({'hr': heart_rate['heart_rate']}),
        data, timestamps = sample_data_uniformly(hr, hr_x, sampling_rate=sampling_frequency_hr, mode="quadratic")
        return timestamps, data

    @staticmethod
    def read_ecg_freq_from_file(signal_headers):
        sampling_freq = [signal['sample_rate'] for signal in signal_headers if signal['label'] == 'ECG']
        assert len(sampling_freq) == 1, f"No or too many ECG signals: {sampling_freq}"
        return sampling_freq[0]

    @staticmethod
    def read_acceleration_freq_from_file(signal_headers):
        sample_rates = list(map(lambda h: h["sample_rate"], filter(lambda h: "Acc" in h["label"], signal_headers)))
        if any([sr != sample_rates[0] for sr in sample_rates]):
            raise UserWarning(f"Not all Faros accelerometer sampling rates are the same: {sample_rates}")
        return sample_rates[0]

    @staticmethod
    def read_acceleration_data_from_file(signals, signal_headers):
        headers = ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']
        data = []
        for column in headers:
            idx = [header['label'] for header in signal_headers].index(column)
            data.append(signals[idx])

        data_body = np.stack(data, axis=1)
        return pd.DataFrame(data_body, columns=headers)

    @staticmethod
    def read_ecg_signal(signals, signal_headers):
        ecg_signal = [signal['label'] for signal in signal_headers].index("ECG")
        return signals[ecg_signal]
