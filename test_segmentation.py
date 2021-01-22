from src.azure import AzureKinect
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import butter, sosfiltfilt
from scipy.signal import find_peaks

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

bjarne = AzureKinect("data/bjarne.csv")
bjarne.process_raw_data()
pelvis = bjarne["pelvis"].to_numpy()[:, 1]


def butterworth(signal, cutoff = 0.1, order = 4):
    sos = butter(order, cutoff, output='sos')
    return sosfiltfilt(sos, signal, axis=0)


def load_example():
    example = AzureKinect("data/jonas.csv")
    example.process_raw_data()
    example = example["pelvis"].to_numpy()[:, 1]
    example = butterworth(example)
    example = example[238:294]
    # plt.plot(example)
    # plt.show()
    return example


def dtw(pelvis):
    # example = pelvis[450:503]
    # l = len(example)
    # costs = []
    #
    # for window in range(len(pelvis) - l + 1):
    #     sliding_window = pelvis[window:window + l]
    #     # print(sliding_window)
    #     distance, path = fastdtw(example, sliding_window)
    #     print(distance)
    #     costs.append(distance)
    #
    # plt.plot(pelvis)
    # plt.plot(costs)
    # plt.show()
    cutoff = 0.1
    order = 4
    sos = butter(order, cutoff, output='sos')
    pelvis = sosfiltfilt(sos, pelvis, axis=0)


def zero_velocity_crossing(pelvis):
    pelvis = (pelvis - pelvis.mean()) / pelvis.std()
    velocity = np.gradient(pelvis)
    zero_crossing = np.where((velocity > -0.01) & (velocity < 0.01))
    # plt.hist(velocity)
    plt.plot(pelvis)
    plt.plot(velocity)
    plt.scatter(zero_crossing, velocity[zero_crossing])
    plt.show()


def peak_finding(pelvis):
    example = load_example()
    # pelvis = (pelvis - pelvis.mean()) / pelvis.std()
    peaks, _ = find_peaks(pelvis * -1, height=0)

    for p1, p2 in zip(peaks, peaks[1:]):
        # if abs(p1 - p2) < 20:
        #     continue

        test = pelvis[p1:p2]
        distance, path = fastdtw(example, test)
        print(f"p1: {p1}, p2: {p2} -> {distance}")

    plt.plot(pelvis)
    plt.plot(example)
    plt.scatter(peaks, pelvis[peaks])
    plt.show()


peak_finding(pelvis)
