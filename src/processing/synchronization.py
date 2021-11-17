from scipy import signal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_cross_correlation(
        reference_signal: np.ndarray,
        target_signal: np.ndarray,
        sampling_frequency: int,
        plot: bool = False,
):
    if plot:
        plt.plot(reference_signal, label=f"Reference Signal")
        plt.plot(target_signal, label=f"Target Signal")
        plt.legend()
        plt.show()

    corr = signal.correlate(reference_signal, target_signal)
    shift_in_samples = np.argmax(corr) - len(target_signal) - 1
    return shift_in_samples / sampling_frequency


def infer_sampling_frequency(series: pd.Series) -> int:
    return int(np.mean(1.0 / np.diff(series.index)))
