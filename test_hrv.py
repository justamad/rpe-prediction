from pyedflib import highlevel
from biosppy.signals import ecg
from rpe_prediction.config import ConfigReader

from scipy import interpolate

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

file_name = "data/bjarne_trial/faros/record.EDF"
signals, signal_headers, header = highlevel.read_edf(file_name)
index = [signal['label'] for signal in signal_headers].index("ECG")
ecg_signal = signals[index]

config = ConfigReader("data/bjarne_trial/config.json")


def uniform_sampling(x_axis, y_axis, new_sampling_rate, mode="cubic"):
    num = int(x_axis[-1] * new_sampling_rate)  # Define new constant sampling points
    xx = np.linspace(x_axis[0], x_axis[-1], num)
    f = interpolate.interp1d(x_axis, y_axis, kind=mode)
    return xx, f(xx)


for entry in config.iterate_over_sets():
    start, end = entry['faros']
    heart_rate = ecg.ecg(ecg_signal[start * 5:end * 5], sampling_rate=500, show=True)
    r_peaks = heart_rate['rpeaks'] / 500
    r_intervals = np.diff(r_peaks)
    x, evenly = uniform_sampling(r_peaks[:-1], r_intervals, 4)
    print(len(r_peaks), len(r_intervals), len(evenly))

    plt.figure(figsize=(12, 4))
    plt.title("RR Intervals Equidistant sampled")
    plt.plot(r_peaks[:-1], r_intervals, label="R peaks")
    plt.plot(x, evenly, label="Resampled")
    plt.xlabel("Time (s)")
    plt.ylabel("RR Interval (s)")
    plt.legend()
    plt.show()
