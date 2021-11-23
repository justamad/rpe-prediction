from scipy.signal import butter, sosfiltfilt
from scipy import interpolate
from numpy import linalg as la

import numpy as np
import pandas as pd
import logging


def calculate_magnitude(
        data: np.ndarray,
        norm: int = 2,
):
    return la.norm(data, ord=norm, axis=1)


def calculate_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    acceleration_data = np.gradient(np.gradient(df.to_numpy(), axis=0), axis=0)
    return pd.DataFrame(acceleration_data, columns=df.columns, index=df.index)


def resample_data(
        df: pd.DataFrame,
        cur_fs: int,
        new_fs: int,
        mode: str = 'cubic',
) -> pd.DataFrame:
    x = np.arange(len(df)) / cur_fs
    num = int(x[-1] * new_fs)  # Define new constant sampling points
    xx = np.linspace(x[0], x[-1], num)

    result_columns = []
    for column in df.columns:
        f = interpolate.interp1d(x, df[column], kind=mode)
        result_columns.append(f(xx))

    return pd.DataFrame(np.array(result_columns).T, columns=df.columns)


def butter_bandpass(fc: int, fs: int, order=5):
    w = fc / fs / 2
    sos = butter(order, w, btype='lowpass', analog=False, output='sos')
    return sos


def butterworth_filter_1d(data: np.ndarray, fs, fc, order=4):
    sos = butter_bandpass(fc=fc, fs=fs, order=order)
    return sosfiltfilt(sos, data)


def normalize_signal(data: np.ndarray):
    return (data - data.mean()) / data.std()


def find_closest_timestamp(timestamps, point):
    differences = np.abs(timestamps - point)
    return np.argmin(differences)


def apply_butterworth_filter(
        df: pd.DataFrame,
        cutoff: int,
        sampling_rate: int,
        order: int = 4,
) -> pd.DataFrame:
    sos = butter_bandpass(cutoff, sampling_rate, order)
    result = []

    for column in df.columns:
        result.append(sosfiltfilt(sos, df[column]))

    return pd.DataFrame(data=np.array(result).T, columns=df.columns, index=df.index)


def identify_and_fill_gaps_in_data(
        df: pd.DataFrame,
        sampling_rate: int,
        method: str = "quadratic",
        log: bool = False,
) -> pd.DataFrame:
    diffs = np.diff(df.index) / (1 / sampling_rate)
    diffs = (np.round(diffs) - 1).astype(np.uint32)
    if log:
        logging.info(f'Number of missing data points: {np.sum(diffs)}')

    df.reset_index(drop=False, inplace=True)
    df_new = pd.DataFrame(columns=df.columns)
    for idx, missing_frames in enumerate(diffs):
        df_new = df_new.append(df.iloc[idx])

        for _ in range(missing_frames):
            df_new = df_new.append(pd.Series(), ignore_index=True)

    return df_new.interpolate(method=method).set_index('timestamp', drop=True)
