from fastdtw import fastdtw
from scipy import signal

import numpy as np
import matplotlib.pyplot as plt


def segment_exercises_based_on_joint(joint_data: np.array, exemplar: np.array, max_cost: int, show: bool):
    """
    Segment data based on a given joint and a given example
    @param joint_data: 1D-trajectory of the target axis
    @param exemplar: exemplar repetition
    @param max_cost: maximum threshold for DTW cost
    @param show: flag if results should be shown immediately
    @return: TODO
    """
    peaks = signal.find_peaks(-joint_data)[0]

    candidates = []
    costs = []

    # Calculate costs for segmentation candidates
    for t1, t2 in zip(peaks, peaks[1:]):
        observation = joint_data[t1:t2]
        cost = calculate_fast_dtw_cost(exemplar, observation)
        costs.append(cost)
        if cost < max_cost:
            candidates.append(observation)

            if show:
                plt.plot(observation, label=len(candidates))

    return candidates, costs


def calculate_fast_dtw_cost(exemplar, observation):
    """
    Calculate the Fast Dynamic Time Warping costs between example and observation
    @param exemplar: the exemplar observation
    @param observation: a new observation
    @return: total DTW cost
    """
    total_cost, warp_path = fastdtw(exemplar, observation)
    return total_cost
