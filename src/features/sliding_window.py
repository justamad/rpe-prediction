from .statistical_features import calculate_range, calculate_std

import pandas as pd

feature_dict = {
    'std': calculate_std,
    'range': calculate_range
}


def calculate_features_sliding_window(df: pd.DataFrame, window_size: int, step_size: int = 1):
    """
    Method calculates features for sliding windows
    @param df: the current dataframe that holds data from a set, repetition or arbitrary timespan
    @param window_size: the desired window size
    @param step_size: the desired step size
    @return: pandas data frame that holds the calculated features in columns for all sensors/joints
    """
    features = {f: [] for f in feature_dict.keys()}

    length = len(df) - window_size + 1
    for window in range(length):
        data = df[window:window + window_size - 1].copy()

        for feature_name, method in feature_dict.items():
            f = method(data)
            features[feature_name].append(f)

    all_features = [pd.concat(features_list).add_prefix(f"{name}_") for name, features_list in features.items()]
    final = pd.concat(all_features, axis=1)
    return final
