from scipy.signal import butter, sosfiltfilt, medfilt
from scipy import interpolate

import numpy as np
import pandas as pd


def upsample_data(data, old_sampling_rate, new_sampling_rate, mode='cubic'):
    x = np.arange(len(data)) / old_sampling_rate
    num = int(x[-1] * new_sampling_rate)  # Define new constant sampling points
    xx = np.linspace(x[0], x[-1], num)
    f = interpolate.interp1d(x, data, kind=mode)
    return f(xx)


def butter_bandpass(fc, fs, order=5):
    w = fc / fs / 2
    sos = butter(order, w, btype='lowpass', analog=False, output='sos')
    return sos


def butterworth_filter_1d(data: np.ndarray, fs, fc, order=4):
    sos = butter_bandpass(fc=fc, fs=fs, order=order)
    return sosfiltfilt(sos, data)


def normalize_signal(data):
    return (data - data.mean()) / data.std()


def find_closest_timestamp(timestamps, point):
    differences = np.abs(timestamps - point)
    return np.argmin(differences)


def apply_butterworth_filter(df: pd.DataFrame, cutoff, sampling_rate, order=4):
    sos = butter_bandpass(cutoff, sampling_rate, order)
    result = []

    for column in range(df.shape[1]):
        result.append(sosfiltfilt(sos, df.iloc[:, column]))

    return pd.DataFrame(data=np.array(result).T, columns=df.columns)


def sample_data_uniformly(df: pd.DataFrame, sampling_rate: int, mode: str = "cubic"):
    timestamps = df['timestamp'].to_numpy()
    nr_samples = int(timestamps[-1] - timestamps[0]) * sampling_rate  # The new number of samples after upsampling
    upsampled_timestamps = np.linspace(timestamps[0], timestamps[-1], nr_samples)

    frames, features = df.shape
    data = df.to_numpy()

    uniform_sampled_data = [upsampled_timestamps]
    for feature in range(1, features):
        y = data[:, feature]
        f = interpolate.interp1d(timestamps, y, kind=mode)
        yy = f(upsampled_timestamps)
        uniform_sampled_data.append(yy)

    return pd.DataFrame(data=np.array(uniform_sampled_data).T, columns=df.columns)


def identify_and_fill_gaps_in_data(df: pd.DataFrame, sampling_rate: int, method: str = "quadratic", log: bool = False):
    diffs = np.diff(df.index) / (1 / sampling_rate)
    diffs = (np.round(diffs) - 1).astype(np.uint32)
    if log:
        print(f'Number of missing data points: {np.sum(diffs)}')

    df.reset_index(drop=False, inplace=True)
    df_new = pd.DataFrame(columns=df.columns)
    for idx, missing_frames in enumerate(diffs):
        df_new = df_new.append(df.iloc[idx])

        for _ in range(missing_frames):
            df_new = df_new.append(pd.Series(), ignore_index=True)

    df_new = df_new.interpolate(method=method)
    return df_new.set_index('timestamp', drop=True)
