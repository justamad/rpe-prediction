from .signal_processing import apply_butterworth_1d_signal
from typing import List, Tuple
from scipy import signal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def segment_kinect_signal(
        input_signal: pd.Series,
        prominence: float,
        std_dev_p: float,
        min_dist_p: float,
        min_time: int,
        mode: str = "full",
        show: bool = False,
        log_path: str = None,
) -> Tuple[List[Tuple], List[Tuple]]:
    if mode not in ["full", "concentric", "eccentric"]:
        raise ValueError(f"Mode must be either 'all', 'ecc' or 'con'. Given '{mode}'.")

    signal_norm = (input_signal.to_numpy() - np.mean(input_signal)) / np.std(input_signal)
    signal_norm = apply_butterworth_1d_signal(signal_norm, cutoff=6, order=4, sampling_rate=30)

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

    if show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title("Found peaks")
        ax1.plot(signal_norm)
        ax1.scatter(peaks, signal_norm[peaks])

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

    full_repetitions = list([(input_signal.index[p0], input_signal.index[p1]) for p0, p1 in valid_peaks])
    if mode == "full":
        return full_repetitions, full_repetitions

    final_peaks = []
    for p0, p1 in valid_peaks:
        max_peak = np.argmax(-signal_norm[p0:p1])
        if mode == "concentric":
            final_peaks.append([p0, p0 + max_peak])
        else:
            final_peaks.append([p0 + max_peak, p1])

    part_repetitions = list([(input_signal.index[p0], input_signal.index[p1]) for p0, p1 in final_peaks])
    return part_repetitions, full_repetitions
