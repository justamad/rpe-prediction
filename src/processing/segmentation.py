from .signal_processing import apply_butterworth_1d_signal
from typing import List, Tuple
from scipy import signal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("WebAgg")


def find_next_peak(orig_signal: np.ndarray, other_peaks: np.ndarray, current_peak: int, min_dist: int) -> int:
    closest_peaks = other_peaks[np.abs(other_peaks - current_peak) < min_dist]
    if len(closest_peaks) == 0:
        return current_peak

    idx = np.argmax(orig_signal[closest_peaks])
    return closest_peaks[idx]


def segment_imu_signal(
        original_series: pd.Series,
        prominence: float,
        std_dev_p: float,
        min_dist_p: float,
        min_time: int,
        show: bool = False,
        log_path: str = None,
) -> List[Tuple]:
    low_frequency = apply_butterworth_1d_signal(original_series.to_numpy(), cutoff=4, order=4, sampling_rate=128)
    low_frequency = (low_frequency - np.mean(low_frequency)) / np.std(low_frequency)
    low_peaks, _ = signal.find_peaks(low_frequency, prominence=prominence)  # , height=0)

    orig_numpy = original_series.to_numpy()
    orig_numpy = (orig_numpy - np.mean(orig_numpy)) / np.std(orig_numpy)
    high_peaks, _ = signal.find_peaks(orig_numpy, prominence=prominence)  # , height=0)

    matched_peaks = []
    for idx, peak in enumerate(low_peaks):
        value = find_next_peak(orig_numpy, high_peaks, peak, 128)
        matched_peaks.append(value)

    plt.plot(low_frequency, label="Low Freq")
    plt.scatter(low_peaks, low_frequency[low_peaks], c="red")
    plt.plot(orig_numpy, label="Original")
    plt.scatter(high_peaks, orig_numpy[high_peaks], c="blue")
    plt.scatter(matched_peaks, orig_numpy[matched_peaks], c="green")#
    plt.legend()
    plt.show()

    valid_peaks = []
    for p1, p2 in zip(matched_peaks, matched_peaks[1:]):
        candidate_segment = orig_numpy[p1:p2]

        if p2 - p1 < min_time:
            continue

        std_dev = np.std(candidate_segment)
        if std_dev < std_dev_p:
            continue

        total_diff = np.abs(np.min(candidate_segment) - np.max(candidate_segment)) * min_dist_p
        end_dist = np.abs(candidate_segment[-1] - candidate_segment[0])
        if end_dist > total_diff:
            continue

        valid_peaks.append([p1, p2])

    t1 = [p[0] for p in valid_peaks]
    t2 = [p[1] for p in valid_peaks]
    final_segments = list(zip(original_series.index[t1], original_series.index[t2]))

    if show:
        plt.close()
        plt.cla()
        plt.clf()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title("Found peaks")
        ax1.plot(low_frequency)
        # ax1.plot(joint_signal, label="Position data")
        # ax1.plot(joint_series.to_numpy(), label="Position data")
        ax1.scatter(low_peaks, low_frequency[low_peaks])

        # ax2.plot(joint_signal)
        # for p1, p2 in final_segments:
            # ax2.plot(joint_series[p1:p2])

        plot_peaks = set([p[0] for p in valid_peaks] + [p[1] for p in valid_peaks])
        plot_peaks = sorted(list(plot_peaks))
        ax2.plot(orig_numpy)
        ax2.set_title(f"Segments: {len(valid_peaks)}")
        ax2.scatter(plot_peaks, orig_numpy[plot_peaks])

        plt.tight_layout()
        if log_path is not None:
            plt.savefig(log_path)
        else:
            plt.show()

        plt.close()
        plt.clf()
        plt.cla()

    return final_segments


def segment_kinect_signal(
        input_signal: pd.Series,
        prominence: float,
        std_dev_p: float,
        min_dist_p: float,
        min_time: int,
        show: bool = False,
        log_path: str = None,
) -> List[Tuple]:
    signal_norm = (input_signal.to_numpy() - np.mean(input_signal)) / np.std(input_signal)
    peaks, _ = signal.find_peaks(signal_norm, prominence=prominence)
    peaks = np.insert(peaks, 0, 0)
    peaks = np.append(peaks, [len(signal_norm) - 1])

    valid_peaks = []
    for p1, p2 in zip(peaks, peaks[1:]):
        candidate_segment = signal_norm[p1:p2]

        if p2 - p1 < min_time:
            continue
        if np.std(candidate_segment) < std_dev_p:
            continue

        total_diff = np.abs(np.min(candidate_segment) - np.max(candidate_segment)) * min_dist_p
        end_dist = np.abs(candidate_segment[-1] - candidate_segment[0])
        if end_dist > total_diff:
            continue

        valid_peaks.append([p1, p2])

    t1 = [p[0] for p in valid_peaks]
    t2 = [p[1] for p in valid_peaks]
    final_segments = list(zip(input_signal.index[t1], input_signal.index[t2]))

    if show:
        plt.close()
        plt.cla()
        plt.clf()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title("Found peaks")
        ax1.plot(signal_norm)
        ax1.scatter(peaks, signal_norm[peaks])

        # ax2.plot(joint_signal)
        # for p1, p2 in final_segments:
            # ax2.plot(joint_series[p1:p2])

        plot_peaks = set([p[0] for p in valid_peaks] + [p[1] for p in valid_peaks])
        plot_peaks = sorted(list(plot_peaks))
        ax2.plot(signal_norm)
        ax2.set_title(f"Segments: {len(valid_peaks)}")
        ax2.scatter(plot_peaks, signal_norm[plot_peaks])

        plt.tight_layout()
        if log_path is not None:
            plt.savefig(log_path)
        else:
            plt.show()

        plt.close()
        plt.clf()
        plt.cla()

    return final_segments


if __name__ == '__main__':
    from src.processing import apply_butterworth_filter
    from os.path import join
    imu_df = pd.read_csv(join("../../data/processed/9AE368/01_set", "pos.csv"), index_col=0)
    imu_df.index = pd.to_datetime(imu_df.index)

    # imu_df_filter = apply_butterworth_filter(df=imu_df, cutoff=10, order=4, sampling_rate=128)
    reps = segment_kinect_signal(
        imu_df["PELVIS (y)"],
        prominence=0.01,
        std_dev_p=0.4,
        min_dist_p=0.5,
        min_time=30,
        show=True,
    )
