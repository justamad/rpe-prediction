from src.processing import apply_butterworth_filter, normalize_signal, find_peaks
from pyedflib import highlevel
from biosppy.signals import ecg

import pandas as pd
import numpy as np
import os


class Faros(object):

    def __init__(self, folder):
        if not os.path.exists(folder):
            raise Exception(f"Folder {folder} does not exist.")

        file_name = os.path.join(folder, "record.edf")
        self.file_name = file_name
        self.acc_data = self.read_acceleration_data_from_file()
        self.ecg_data = self.read_ecg_signal()
        self._sampling_frequency = self.read_acceleration_freq_from_file()
        self._sampling_frequency_ecg = self.read_ecg_freq_from_file()

        # Peak finding parameters
        self.height = 1.2
        self.prominence = 1.0
        self.distance = None

    def get_acceleration_data(self):
        data_np = self.acc_data.to_numpy()
        return data_np

    def get_synchronization_signal(self) -> np.ndarray:
        data = self.acc_data['Accelerometer_X'].to_numpy()
        return data

    def get_timestamps(self) -> np.ndarray:
        return np.arange(len(self.acc_data)) / self.sampling_frequency

    def get_synchronization_data(self):
        clock = self.get_timestamps()
        raw_signal = apply_butterworth_filter(self.get_synchronization_signal())
        raw_signal = normalize_signal(raw_signal)
        processed_signal = -raw_signal  # Inversion of coordinate system
        peaks = find_peaks(-processed_signal, height=self.height, prominence=self.prominence)
        return clock, raw_signal, processed_signal, peaks

    def get_heart_rate_signal(self):
        heart_rate = ecg.ecg(self.ecg_data, sampling_rate=self._sampling_frequency_ecg, show=False)
        return heart_rate['heart_rate_ts'], heart_rate['heart_rate']

    def read_acceleration_freq_from_file(self):
        _, signal_headers, _ = highlevel.read_edf(self.file_name)
        sample_rates = list(map(lambda h: h["sample_rate"], filter(lambda h: "Acc" in h["label"], signal_headers)))
        if any([sr != sample_rates[0] for sr in sample_rates]):
            raise UserWarning(f"Not all Faros accelerometer sampling rates are the same: {sample_rates}")

        return sample_rates[0]

    def read_ecg_freq_from_file(self):
        _, signal_header, _ = highlevel.read_edf(self.file_name)
        sampling_freq = [signal['sample_rate'] for signal in signal_header if signal['label'] == 'ECG']
        assert len(sampling_freq) == 1, f"No or too many ECG signals: {sampling_freq}"
        return sampling_freq[0]

    def read_acceleration_data_from_file(self):
        signals, signal_headers, header = highlevel.read_edf(self.file_name)

        headers = ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']
        data = []
        for column in headers:
            idx = [header['label'] for header in signal_headers].index(column)
            data.append(signals[idx])

        data_body = np.stack(data, axis=1)
        return pd.DataFrame(data_body, columns=headers)

    def read_ecg_signal(self):
        signals, signal_headers, _ = highlevel.read_edf(self.file_name)
        ecg_signal = [signal['label'] for signal in signal_headers].index("ECG")
        return signals[ecg_signal]

    def get_signal_from_datasource(self, label):
        signals, signal_headers, header = highlevel.read_edf(self.file_name)
        idx = [header['label'] for header in signal_headers].index(label)
        return signals[idx]

    def cut_trial(self, start_idx, end_idx):
        data = self.acc_data.iloc[start_idx:end_idx]
        return data

    @property
    def sampling_frequency(self):
        return self._sampling_frequency

    def __repr__(self):
        return "Faros"
