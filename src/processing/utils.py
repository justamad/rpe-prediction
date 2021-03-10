from scipy import signal
from scipy import interpolate

import numpy as np
import pandas as pd


def upsample_data(data, old_sampling_rate, new_sampling_rate, mode="cubic"):
    x = np.arange(len(data)) / old_sampling_rate
    num = int(x[-1] * new_sampling_rate)  # Define new constant sampling points
    xx = np.linspace(x[0], x[-1], num)
    f = interpolate.interp1d(x, data, kind=mode)
    return f(xx)


def find_peaks(data, height, prominence, distance=None):
    peaks, _ = signal.find_peaks(data, height=height, prominence=prominence, distance=distance)
    return peaks


def apply_butterworth_filter(data, order=4, cut_off=0.1):
    sos = signal.butter(order, cut_off, output='sos')
    return signal.sosfiltfilt(sos, data)


def normalize_signal(data):
    return (data - data.mean()) / data.std()


def apply_butterworth_filter_dataframe(data_frame, sampling_frequency):
    """
    Applies a butterworth filter to given data array
    :param data_frame: pandas data frame consisting the data
    :param sampling_frequency: the current sampling frequency
    :return: pandas array with butterworth-filtered data
    """
    fc = 6  # Cut-off frequency of the filter
    w = fc / (sampling_frequency / 2)  # Normalize the frequency
    b, a = signal.butter(4, w, 'lp', analog=False)

    data = data_frame.to_numpy()
    rows, cols = data.shape
    result = []

    for column in range(cols):
        raw_signal = data[:, column]
        filtered_signal = signal.lfilter(b, a, raw_signal)
        result.append(filtered_signal)

    return pd.DataFrame(data=np.array(result).T, columns=data_frame.columns)


def sample_data_uniformly(data_frame, timestamps, sampling_rate, mode="cubic"):
    """
    Applies a uniform sampling to given data frame
    :param data_frame: data frame consisting the data
    :param timestamps: timestamps for given data, in seconds
    :param sampling_rate: desired sampling frequency in frames per seconds (or Hz)
    :param mode: the way of interpolating the data [linear, quadratic, cubic, ..]
    :return: data frame with filtered data
    """
    nr_samples = int(timestamps[-1] - timestamps[0]) * sampling_rate  # The new number of samples after upsampling
    upsampled_timestamps = np.linspace(timestamps[0], timestamps[-1], nr_samples)

    frames, features = data_frame.shape
    data = data_frame.to_numpy()

    uniform_sampled_data = []
    for feature in range(features):
        y = data[:, feature]
        f = interpolate.interp1d(timestamps, y, kind=mode)
        yy = f(upsampled_timestamps)
        uniform_sampled_data.append(yy)

    return pd.DataFrame(data=np.array(uniform_sampled_data).T, columns=data_frame.columns), upsampled_timestamps


def fill_missing_data(data, delta=33333):
    """
   Applies a uniform sampling to given data frame
   :param data: data frame consisting the data
   :param delta: desired sampling frequency in microseconds
   :return: data frame with filtered data
   """
    _, cols = data.shape
    data_body = data.to_numpy()
    diffs = np.diff(data["timestamp"]) / delta
    diffs = (np.round(diffs) - 1).astype(np.uint32)
    print(f'Number of missing data points: {np.sum(diffs)}')

    inc = 0
    for idx, missing_frames in enumerate(diffs):
        if missing_frames <= 0:
            continue

        for j in range(missing_frames):
            data_body = np.insert(data_body, idx + inc + j + 1, np.full(cols, np.nan), axis=0)

        inc += missing_frames

    data = pd.DataFrame(data_body, columns=data.columns).interpolate(method='quadratic')
    return data
