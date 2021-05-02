from tsfresh.feature_extraction import MinimalFCParameters
from src.processing import get_joints_as_list

import tsfresh
import pandas as pd
import math

settings = MinimalFCParameters()


def calculate_features_sliding_window(df: pd.DataFrame, window_size: int, step_size: int = 1):
    """
    Method calculates features for sliding windows
    @param df: the current dataframe that holds data from a set, repetition or arbitrary timespan
    @param window_size: the desired window size
    @param step_size: the desired step size
    @return: pandas data frame that holds the calculated features in columns for all sensors/joints
    """
    windows = []
    n_windows = math.floor((len(df) - window_size) / step_size) + 1

    print(f"Calculated Windows: {n_windows}")
    for window_id, window_index in enumerate(range(0, n_windows, step_size)):
        window = df[window_index:window_index + window_size - 1].copy()
        window['id'] = window_id
        windows.append(window)

    df = pd.concat(windows, ignore_index=True)
    return calculate_ts_features(df)


def calculate_ts_features(df):
    """
    Calculate features for entire dataframe that contains window index with column 'id'
    @param df: dataframe for all possible windows
    @return: calculated features
    """
    feat = tsfresh.extract_features(df, column_id='id', column_sort='timestamp', default_fc_parameters=settings)
    return feat
