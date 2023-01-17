from .signal_processing import resample_data, normalize_signal
from scipy import signal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md


def calculate_cross_correlation_with_datetime(
        reference_df: pd.DataFrame,
        ref_sync_axis: str,
        target_df: pd.DataFrame,
        target_sync_axis: str,
        show: bool = False,
        log_path: str = None,
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
    )

    shift_dt = (reference_df.index[0] - target_df.index[0]) + pd.Timedelta(seconds=shift_seconds)

    temp = target_df.copy()
    temp.index = temp.index + (reference_df.index[0] - target_df.index[0])

    temp2 = target_df.copy()
    temp2.index = temp2.index + shift_dt

    if show:
        xfmt = md.DateFormatter("%M:%S")

        plt.close()
        fix, axs = plt.subplots(2, 1)
        axs[0].plot(reference_df[ref_sync_axis], label=f"Phyislog")
        axs[0].plot(temp[target_sync_axis], label=f"Azure Kinect")
        axs[1].plot(reference_df[ref_sync_axis], label=f"Phyiolog")
        axs[1].plot(temp2[target_sync_axis], label=f"Azure Kinect")

        axs[0].xaxis.set_major_formatter(xfmt)
        axs[1].xaxis.set_major_formatter(xfmt)

        plt.legend()
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Acceleration")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Acceleration")

        plt.tight_layout()
        # plt.savefig("sync.pdf", dpi=300)
        if log_path is not None:
            plt.savefig(log_path)
        else:
            plt.show(block=True)

    return shift_dt


def calculate_cross_correlation(
        reference_signal: np.ndarray,
        target_signal: np.ndarray,
        sampling_frequency: int,
) -> float:
    reference_signal_norm = normalize_signal(reference_signal)
    target_signal_norm = normalize_signal(target_signal)

    corr = signal.correlate(reference_signal_norm, target_signal_norm)
    shift_in_samples = np.argmax(corr) - len(target_signal) - 1

    return shift_in_samples / sampling_frequency


def infer_sampling_frequency(series: pd.DatetimeIndex) -> int:
    diffs = series[1:] - series[:-1]
    total_seconds = diffs.total_seconds()
    diffs = 1.0 / total_seconds
    mean_diff = np.mean(diffs)
    return int(mean_diff + 0.5)
