from fastdtw import fastdtw
from scipy import signal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors


def segment_exercises_based_on_joint(joint_data: np.array, exemplar: np.array, min_duration: int,
                                     std_dev_percentage: float, show: bool = False, path: str = None):
    """
    Segment data based on a given joint and a given example
    @param joint_data: 1D-trajectory of the target axis
    @param exemplar: exemplar repetition
    @param min_duration: the minimum duration of a single repetition
    @param std_dev_percentage: a percentage value
    @param show: flag if results should be shown or saved
    @param path: path to save the image
    @return: list of tuples with start and end points of candidates, list of costs for all observations
    """
    peaks, _ = signal.find_peaks(joint_data, height=1, prominence=1, width=min_duration)
    peaks = [0] + list(peaks) + [len(joint_data) - 1]  # Add first and last point as additional candidates

    candidates = []
    costs = []
    exemplar_std_threshold = np.std(exemplar) * std_dev_percentage

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 10))
    ax1.plot(joint_data, label="Position data")
    ax1.scatter(peaks, joint_data[peaks])

    # Calculate costs for segmentation candidates
    for t1, t2 in zip(peaks, peaks[1:]):
        observation = joint_data[t1:t2]
        # Step 1: first check basic statistics
        std_dev = np.std(observation)
        length = t2 - t1

        if std_dev < exemplar_std_threshold or length < min_duration:
            ax2.plot(np.arange(t1, t2), observation, '--', color="gray")
            continue
        ax2.plot(np.arange(t1, t2), observation)

        # Step 2: check DTW cost
        cost = calculate_fast_dtw_cost(exemplar, observation)
        costs.append(cost)
        candidates.append((t1, t2))

    fig.suptitle(f'Segmentation costs: mean: {np.mean(costs):.2f}, std: {np.std(costs):.2f}')
    for counter, ((t1, t2), cost) in enumerate(zip(candidates, costs)):
        hsv_color = matplotlib.colors.hsv_to_rgb([counter / len(candidates) * 0.75, 1, 1])
        ax3.plot(joint_data[t1:t2], label=f"{counter + 1}: {t1}-{t2}, c={cost:.1f}", color=hsv_color)
    ax3.legend()

    if show:
        if path is not None:
            plt.savefig(path)
        else:
            plt.show()

    plt.close()
    plt.cla()
    plt.clf()

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
