from fastdtw import fastdtw
from scipy import signal
from .utils import get_hsv_color_interpolation

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def segment_1d_joint_on_example(joint_data: pd.Series, exemplar: np.array, min_duration: int, std_dev_percentage: float,
                                show: bool = False, path: str = None):
    """
    Segment data based on a given joint and a given example
    @param joint_data: 1D-trajectory of the target axis, pandas series
    @param exemplar: exemplar repetition
    @param min_duration: the minimum duration of a single repetition
    @param std_dev_percentage: the percentage value multiplied by std dev for example as threshold
    @param show: flag if results should be shown or saved
    @param path: path to save the image
    @return: list of tuples with start and end points of candidates, list of costs for all observations
    """
    # Normalize exemplar and joint data for DTW costs
    exemplar = (exemplar - np.mean(exemplar)) / np.std(exemplar)
    joint_data = (joint_data - np.mean(joint_data)) / np.std(joint_data)

    peaks, _ = signal.find_peaks(joint_data, height=0, prominence=0.05, width=min_duration)

    candidates = []
    dtw_costs = []
    exemplar_std_threshold = np.std(exemplar) * std_dev_percentage

    if show:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 10))
        ax1.plot(joint_data, label="Position data")
        ax1.scatter(peaks, joint_data.to_numpy()[peaks])
        ax4.plot(exemplar, label="Exemplar")

    # Calculate costs for segmentation candidates
    for p1, p2 in zip(peaks, peaks[1:]):
        observation = joint_data[p1:p2]

        # Step 1: first check basic statistics
        if np.std(observation) < exemplar_std_threshold:
            if show:
                ax2.plot(np.arange(p1, p2), observation, '--', color="gray")
            continue

        if show:
            ax2.plot(np.arange(p1, p2), observation)

        # Step 2: check DTW cost
        dtw_costs.append(calculate_fast_dtw_cost(exemplar, observation))
        t1, t2 = joint_data.index[p1], joint_data.index[p2]
        candidates.append((t1, t2))

    if show:
        fig.suptitle(f'DTW Costs: mean: {np.mean(dtw_costs):.2f}, std: {np.std(dtw_costs):.2f}')
        for counter, ((t1, t2), dtw_cost) in enumerate(zip(candidates, dtw_costs)):
            hsv_color = get_hsv_color_interpolation(counter, len(candidates))
            ax3.plot(joint_data.loc[t1:t2], label=f"{counter + 1}: {t1}-{t2}, c={dtw_cost:.1f}", color=hsv_color)
        ax3.legend()

        if path is not None:
            plt.savefig(path)
        else:
            plt.show()

        plt.close()
        plt.cla()
        plt.clf()

    return candidates


def calculate_fast_dtw_cost(exemplar, observation):
    """
    Calculate the Fast Dynamic Time Warping costs between example and observation
    @param exemplar: the exemplar observation
    @param observation: a new observation
    @return: total DTW cost
    """
    total_cost, warp_path = fastdtw(exemplar, observation)
    return total_cost
