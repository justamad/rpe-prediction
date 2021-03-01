from scipy import signal
from scipy import interpolate

import numpy as np
import pandas as pd


def find_peaks(data, height, prominence, distance=None):
    peaks, _ = signal.find_peaks(data, height=height, prominence=prominence, distance=distance)
    return peaks


def apply_butterworth_filter(data, order=4, cut_off=0.1):
    sos = signal.butter(order, cut_off, output='sos')
    return signal.sosfiltfilt(sos, data)


def normalize_signal(data):
    return (data - data.mean()) / data.std()


def upsample_data(data, old_sampling_rate, new_sampling_rate, mode="cubic"):
    x = np.arange(len(data)) / old_sampling_rate
    num = int(x[-1] * new_sampling_rate)  # Define new constant sampling points
    xx = np.linspace(x[0], x[-1], num)
    f = interpolate.interp1d(x, data, kind=mode)
    return f(xx)


def add_magnitude(data: np.ndarray) -> np.ndarray:
    assert data.shape[1] < 10, "You probably flipped an array, here"
    result = np.empty((data.shape[0], data.shape[1] + 1))
    result[:, :-1] = data
    result[:, -1] = np.sqrt(np.square(data).sum(axis=1))
    return result


def sample_data_uniformly(data_frame, sampling_rate):
    """
    Applies a uniform sampling to given data frame
    :param data_frame: data frame consisting the data
    :param sampling_rate: desired sampling frequency
    :return: data frame with filtered data
    """
    timestamps = data_frame['timestamp'].to_numpy()
    x = timestamps - timestamps[0]  # shift to zero

    # Define new constant sampling points
    num = int(x[-1] * sampling_rate)  # 30 fps
    xx = np.linspace(x[0], x[-1], num)

    frames, features = data_frame.shape
    data = data_frame.to_numpy()

    uniform_sampled_data = []
    for feature in range(features):
        y = data[:, feature]
        f = interpolate.interp1d(x, y, kind="cubic")
        yy = f(xx)
        uniform_sampled_data.append(yy)

    return pd.DataFrame(data=np.array(uniform_sampled_data).T, columns=data_frame.columns)


def fill_missing_data(data, delta=33333):
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
