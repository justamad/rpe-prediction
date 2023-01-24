from typing import List, Tuple
from scipy import signal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def segment_signal_peak_detection(
        joint_series: pd.Series,
        prominence: float = 0.1,
        std_dev_p: float = 0.5,
        show: bool = False,
        log_path: str = None,
) -> List[Tuple]:
    joint_signal = (joint_series.to_numpy() - np.mean(joint_series)) / np.std(joint_series)
    peaks, _ = signal.find_peaks(joint_signal, prominence=prominence)  # , height=0)

    valid_peaks = []
    for p1, p2 in zip(peaks, peaks[1:]):
        std_dev = np.std(joint_signal[p1:p2])
        if std_dev > std_dev_p:
            valid_peaks.append([p1, p2])

    if show:
        plt.close()
        plt.cla()
        plt.clf()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title("Found peaks")
        ax1.plot(joint_signal, label="Position data")
        ax1.plot(joint_series.to_numpy(), label="Position data")
        ax1.scatter(peaks, joint_signal[peaks])

        ax2.set_title("Valid peaks")
        ax2.plot(joint_signal)
        plot_peaks = set([p[0] for p in valid_peaks] + [p[1] for p in valid_peaks])
        plot_peaks = sorted(list(plot_peaks))
        ax2.scatter(plot_peaks, joint_signal[plot_peaks])

        plt.tight_layout()
        if log_path is not None:
            plt.savefig(log_path)
        else:
            plt.show()

        plt.close()
        plt.clf()
        plt.cla()

    t1 = [p[0] for p in valid_peaks]
    t2 = [p[1] for p in valid_peaks]
    final_segments = list(zip(joint_series.index[t1], joint_series.index[t2]))
    return final_segments
