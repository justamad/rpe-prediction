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


def apply_butterworth_filter(data, order=4, cut_off=0.1):
    sos = signal.butter(order, cut_off, output='sos')
    return signal.sosfiltfilt(sos, data)


def normalize_signal(data):
    return (data - data.mean()) / data.std()


def find_closest_timestamp(timestamps, point):
    differences = np.abs(timestamps - point)
    return np.argmin(differences)


def apply_butterworth_df(data_frame, sampling_frequency, order=4, fc=6):
    """
    Applies a Butterworth filter to the given dataframe
    @param data_frame: data frame that contains positional, orientation or acceleration data
    @param sampling_frequency: the current sampling frequency
    @param order: The order of the Butterworth filter
    @param fc: the cut-off frequency used in Butterworth filter
    @return: data frame with filtered data
    """
    w = fc / (sampling_frequency / 2)  # Normalize the frequency
    b, a = signal.butter(order, w, 'lp', analog=False)

    data = data_frame.to_numpy()
    rows, cols = data.shape
    result = []

    for column in range(cols):
        raw_signal = data[:, column]
        filtered_signal = signal.lfilter(b, a, raw_signal)
        result.append(filtered_signal)

    return pd.DataFrame(data=np.array(result).T, columns=data_frame.columns)


def sample_data_uniformly(data_frame, sampling_rate, mode="cubic"):
    """
    Applies a uniform sampling to given data frame
    :param data_frame: data frame consisting the data
    :param sampling_rate: desired sampling frequency in frames per seconds (or Hz)
    :param mode: the way of interpolating the data [linear, quadratic, cubic, ..]
    :return: data frame with filtered data
    """
    timestamps = data_frame['timestamp'].to_numpy()
    nr_samples = int(timestamps[-1] - timestamps[0]) * sampling_rate  # The new number of samples after upsampling
    upsampled_timestamps = np.linspace(timestamps[0], timestamps[-1], nr_samples)

    frames, features = data_frame.shape
    data = data_frame.to_numpy()

    uniform_sampled_data = [upsampled_timestamps]
    for feature in range(1, features):
        y = data[:, feature]
        f = interpolate.interp1d(timestamps, y, kind=mode)
        yy = f(upsampled_timestamps)
        uniform_sampled_data.append(yy)

    return pd.DataFrame(data=np.array(uniform_sampled_data).T, columns=data_frame.columns)


def fill_missing_data(data: pd.DataFrame, sampling_frequency: int, method: str = "quadratic", log: bool = False):
    """
    Identifies missing data points ina given data frame and fills it using interpolation
    @param data: the dataframe
    @param sampling_frequency: the current sampling frequency
    @param method: interpolation methods, such as quadratic
    @param log: flag if missing points should be printed
    @return: data frame with filled gaps
    """
    _, cols = data.shape
    data_body = data.to_numpy()
    delta = 1 / sampling_frequency
    diffs = np.diff(data["timestamp"]) / delta
    diffs = (np.round(diffs) - 1).astype(np.uint32)
    if log:
        print(f'Number of missing data points: {np.sum(diffs)}')

    inc = 0
    for idx, missing_frames in enumerate(diffs):
        if missing_frames <= 0:
            continue

        for j in range(missing_frames):
            data_body = np.insert(data_body, idx + inc + j + 1, np.full(cols, np.nan), axis=0)

        inc += missing_frames

    data = pd.DataFrame(data_body, columns=data.columns).interpolate(method=method)
    return data
