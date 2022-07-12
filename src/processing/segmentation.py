from fastdtw import fastdtw
from typing import List
from scipy import signal
from .utils import get_hsv_color

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def segment_1d_joint_on_example(
        joint_series: pd.Series,
        exemplar: np.array,
        std_dev_p: float,
        show: bool = False,
        log_path: str = None,
) -> List:
    # exemplar = (exemplar - np.mean(exemplar)) / np.std(exemplar)
    joint_series = (joint_series - np.mean(joint_series)) / np.std(joint_series)
    peaks, _ = signal.find_peaks(joint_series, height=0, prominence=0.05)

    final_peaks = []
    plot_peaks = []
    for p1, p2 in zip(peaks, peaks[1:]):
        std_dev = np.std(joint_series[p1:p2])
        if std_dev > 0.6:
            t1, t2 = joint_series.index[p1], joint_series.index[p2]
            final_peaks.append((t1, t2))
            plot_peaks.append(t1)
            plot_peaks.append(t2)

    candidates = []
    # dtw_costs = []
    # exemplar_std_threshold = np.std(exemplar) * std_dev_p

    # Check conditions for new candidates
    # for p1, p2 in zip(peaks, peaks[1:]):
    #     observation = joint_series.iloc[p1:p2]
    #
    #     if np.std(observation) < exemplar_std_threshold:
    #         continue
    #
    #     total_cost, warp_path = fastdtw(exemplar, observation)
    #     dtw_costs.append(total_cost)
    #     candidates.append((joint_series.index[p1], joint_series.index[p2]))
    #
    # # Filter out candidates
    # dtw_median = np.median(dtw_costs)
    # t = dtw_median * 1.8

    # final_segments = list(map(lambda x: x[1], filter(lambda x: x[0] < t, zip(dtw_costs, candidates))))
    timestamps = joint_series.index[peaks]
    final_segments = list(zip(timestamps, timestamps[1:]))

    if show:
        plt.close()
        plt.cla()
        plt.clf()

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
        # fig.suptitle(f'DTW: M={np.mean(dtw_costs):.1f}, SD={np.std(dtw_costs):.1f}, MD={dtw_median:.1f}, T={t:.1f}')

        # First plot - exemplar
        ax1.plot(exemplar, label="Exemplar")

        # Second plot - peaks
        ax2.plot(joint_series.to_numpy(), label="Position data")
        ax2.scatter(peaks, joint_series.to_numpy()[peaks])

        ax3.plot(joint_series)
        ax3.scatter(plot_peaks, joint_series[plot_peaks])

        # Third plot - candidates
        # for counter, ((t1, t2), dtw_cost) in enumerate(zip(candidates, dtw_costs)):
        #     ax3.plot(joint_series.loc[t1:t2], label=f"{counter + 1}: dtw={dtw_cost:.1f}",
        #              color=get_hsv_color(counter, len(candidates)))
        # ax3.legend()

        # Third plot - candidates
        # for counter, (t1, t2) in enumerate(final_segments):
        #     ax4.plot(joint_series.loc[t1:t2], color=get_hsv_color(counter, len(final_segments)))

        plt.tight_layout()
        if log_path is not None:
            plt.savefig(log_path)
        else:
            plt.show()

        plt.close()
        plt.clf()
        plt.cla()

    return final_segments
