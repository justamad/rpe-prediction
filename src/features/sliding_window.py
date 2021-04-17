from .statistical_features import calculate_range, calculate_std

import pandas as pd
import numpy as np


def calculate_features_sliding_window(df: pd.DataFrame, window_size: int, step_size: int = 1):
    features = []
    length = len(df) - window_size + 1
    for window in range(length):
        data = df[window:window + window_size - 1].copy()
        f = calculate_std(data)
        features.append(f)

    return pd.concat(features)
