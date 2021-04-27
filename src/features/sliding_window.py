from tsfresh.feature_extraction import ComprehensiveFCParameters
from src.processing import get_joints_as_list

import tsfresh
import pandas as pd

settings = ComprehensiveFCParameters()


def calculate_features_sliding_window(df: pd.DataFrame, window_size: int, step_size: int = 1):
    """
    Method calculates features for sliding windows
    @param df: the current dataframe that holds data from a set, repetition or arbitrary timespan
    @param window_size: the desired window size
    @param step_size: the desired step size
    @return: pandas data frame that holds the calculated features in columns for all sensors/joints
    """
    joints = get_joints_as_list(df, " (x) pos")
    joints.remove('t')  # TODO: Change this

    length = len(df) - window_size + 1
    for window in range(length):
        data = df[window:window + window_size - 1].copy()
        data = reshape_data_for_ts(data, joints)
        feat = tsfresh.extract_features(data, column_id='id', column_sort='timestamp', default_fc_parameters=settings)
        print(feat)

