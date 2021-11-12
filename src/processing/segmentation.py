from fastdtw import fastdtw
from scipy import signal
from .utils import get_hsv_color

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def segment_1d_joint_on_example(joint_data: pd.Series, exemplar: np.array, std_dev_p: float, show: bool = False):
    exemplar = (exemplar - np.mean(exemplar)) / np.std(exemplar)
    joint_data = (joint_data - np.mean(joint_data)) / np.std(joint_data)
    peaks, _ = signal.find_peaks(joint_data, height=0, prominence=0.05)

    candidates = []
    dtw_costs = []
    exemplar_std_threshold = np.std(exemplar) * std_dev_p

    # Check conditions for new candidates
    for p1, p2 in zip(peaks, peaks[1:]):
        observation = joint_data.iloc[p1:p2]

        if np.std(observation) < exemplar_std_threshold:
            continue

        total_cost, warp_path = fastdtw(exemplar, observation)
        dtw_costs.append(total_cost)
        candidates.append((joint_data.index[p1], joint_data.index[p2]))

    # Filter out candidates
    dtw_median = np.median(dtw_costs)
    t = dtw_median * 1.8

    final_segments = list(map(lambda x: x[1], filter(lambda x: x[0] < t, zip(dtw_costs, candidates))))

    if show:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
        fig.suptitle(f'DTW: M={np.mean(dtw_costs):.1f}, SD={np.std(dtw_costs):.1f}, MD={dtw_median:.1f}, T={t:.1f}')

        # First plot - exemplar
        ax1.plot(exemplar, label="Exemplar")

        # Second plot - peaks
        ax2.plot(joint_data.to_numpy(), label="Position data")
        ax2.scatter(peaks, joint_data.to_numpy()[peaks])

        # Third plot - candidates
        for counter, ((t1, t2), dtw_cost) in enumerate(zip(candidates, dtw_costs)):
            ax3.plot(joint_data.loc[t1:t2], label=f"{counter + 1}: dtw={dtw_cost:.1f}",
                     color=get_hsv_color(counter, len(candidates)))
        ax3.legend()

        # Third plot - candidates
        for counter, (t1, t2) in enumerate(final_segments):
            ax4.plot(joint_data.loc[t1:t2], color=get_hsv_color(counter, len(final_segments)))

        plt.tight_layout()
        plt.show()
        plt.close()
        plt.cla()
        plt.clf()

    return final_segments
