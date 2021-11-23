from .signal_processing import resample_data
from scipy import signal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_cross_correlation_with_datetime(
        reference_df: pd.DataFrame,
        ref_sync_axis: str,
        target_df: pd.DataFrame,
        target_sync_axis: str,
        show: bool = False,
):
    reference_fs = infer_sampling_frequency(reference_df.index)
    target_fs = infer_sampling_frequency(target_df.index)

    upsampled_target_df = resample_data(
        df=target_df,
        cur_fs=target_fs,
        new_fs=reference_fs,
    )

    shift_seconds = calculate_cross_correlation(
        reference_signal=reference_df[ref_sync_axis].to_numpy(),
        target_signal=upsampled_target_df[target_sync_axis].to_numpy(),
        sampling_frequency=reference_fs,
        show=False,
    )

    shift_dt = (reference_df.index[0] - target_df.index[0]) + pd.Timedelta(seconds=shift_seconds)
    return shift_dt


def calculate_cross_correlation(
        reference_signal: np.ndarray,
        target_signal: np.ndarray,
        sampling_frequency: int,
        show: bool = False,
):
    if show:
        plt.plot(reference_signal, label=f"Reference Signal")
        plt.plot(target_signal, label=f"Target Signal")
        plt.legend()
        plt.show()

    corr = signal.correlate(reference_signal, target_signal)
    shift_in_samples = np.argmax(corr) - len(target_signal) - 1
    return shift_in_samples / sampling_frequency


def infer_sampling_frequency(series: pd.DatetimeIndex) -> int:
    diffs = series[1:] - series[:-1]
    total_seconds = diffs.total_seconds()
    return int(np.mean(1.0 / total_seconds))
