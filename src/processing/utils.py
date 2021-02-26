from scipy import signal
from scipy import interpolate

import numpy as np


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
