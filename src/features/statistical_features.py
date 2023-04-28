from tsfresh.feature_extraction import ComprehensiveFCParameters, feature_calculators
from typing import List, Tuple

import pandas as pd
import math


class CustomFeatures(ComprehensiveFCParameters):

    def __init__(self):
        ComprehensiveFCParameters.__init__(self)

        for f_name, f in feature_calculators.__dict__.items():
            is_minimal = (hasattr(f, "minimal") and getattr(f, "minimal"))
            is_curtosis_or_skew = f_name == "kurtosis" or f_name == "skewness"
            if f_name in self and not is_minimal and not is_curtosis_or_skew:
                del self[f_name]

        del self["sum_values"]
        del self["variance"]
        del self["mean"]


def apply_sliding_window_time_series(df: pd.DataFrame, overlap: float, win_size: int) -> Tuple[pd.DataFrame, List]:
    windows = []
    labels_majorities = []
    n_windows, stride = calculate_window_parameters(len(df), win_size, overlap)
    for window_idx in range(n_windows):
        window = df.iloc[window_idx * stride:window_idx * stride + win_size].copy()
        reps = list(window["Repetition"])
        labels_majorities.append(max(reps, key=reps.count))
        windows.append(window)

    occurrences = [labels_majorities.count(rep) for rep in list(df["Repetition"].unique())]
    return pd.concat(windows, ignore_index=True), occurrences


def calculate_window_parameters(length: float, window_size: float, overlap: float) -> Tuple[int, int]:
    stride = int(window_size - (window_size * overlap))
    n_windows = math.floor((length - window_size) / stride) + 1
    return n_windows, stride
