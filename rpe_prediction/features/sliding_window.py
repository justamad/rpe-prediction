from tsfresh.feature_extraction import MinimalFCParameters

import tsfresh
import pandas as pd
import math

settings = MinimalFCParameters()
del settings['variance']  # Variance and standard deviation are highly correlated but std integrates nr of samples
del settings['length']  # Length is constant for all windows


def calculate_features_sliding_window(df: pd.DataFrame, window_size: int, overlap: float = 0.25):
    """
    Method calculates features for sliding windows
    @param df: the current dataframe that holds data from a set, repetition or arbitrary timespan
    @param window_size: the desired window size
    @param overlap: the current overlap of sliding windows in percent
    @return: pandas data frame that holds the calculated features in columns for all sensors/joints
    """
    windows = []
    stride = int(window_size - (window_size * overlap))
    n_windows = math.floor((len(df) - window_size) / stride) + 1

    for window_id, window_index in enumerate(range(0, n_windows, stride)):
        window = df[window_index:window_index + window_size - 1].copy()
        window['id'] = window_id
        windows.append(window)

    df = pd.concat(windows, ignore_index=True)
    features = tsfresh.extract_features(df, column_id='id', column_sort='timestamp', default_fc_parameters=settings)
    return features
