from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
from typing import List, Tuple

import datetime
import tsfresh
import pandas as pd
import math

# settings = EfficientFCParameters()
# del settings['variance']  # Variance and standard deviation are highly correlated but std integrates nr of samples
# del settings['length']  # Length is constant for all windows
# del settings['sum_values']  # Highly correlated with RMS and Mean
# del settings['mean']  # Highly correlated with RMS and Sum
# del settings['variance_larger_than_standard_deviation']
# del settings['has_duplicate_max']
# del settings['has_duplicate_min']
# del settings['has_duplicate']
# settings = {k: v for k, v in settings.items() if not isinstance(v, list)}

settings = MinimalFCParameters()
del settings['variance']  # Variance and standard deviation are highly correlated but std integrates nr of samples
del settings['length']  # Length is constant for all windows
del settings['sum_values']  # Highly correlated with RMS and Mean
del settings['mean']  # Highly correlated with RMS and Sum


def calculate_statistical_features_with_sliding_window(
        df: pd.DataFrame,
        window_size: int,
        overlap: float = 0.25,
) -> pd.DataFrame:
    windows = []

    n_windows, stride = calculate_window_parameters(len(df), window_size, overlap)
    for window_idx in range(n_windows):
        window = df.iloc[window_idx * stride:window_idx * stride + window_size].copy()
        window["id"] = window_idx
        windows.append(window)

    df = pd.concat(windows, ignore_index=True)
    features = tsfresh.extract_features(
        timeseries_container=df,
        column_id="id",
        column_sort="timestamp",
        default_fc_parameters=settings,
    )

    features = impute(features)  # Replace Nan and inf by with extreme values (min, max)
    return features


def calculate_statistical_features_with_sliding_window_time_based(
        df_list: List[pd.DataFrame],
        window_size: int,
        overlap: float,
) -> pd.DataFrame:
    origin = datetime.datetime(1970, 1, 1)
    min_start = min([df.index[0] for df in df_list])
    delta = origin - min_start

    for df in df_list:
        df.index = df.index + delta

    max_time = max([df.index[-1] for df in df_list])
    length = (max_time - origin).total_seconds() * 1e3
    window_size = window_size * 1e3
    n_windows, stride = calculate_window_parameters(length, window_size, overlap)

    stride = datetime.timedelta(milliseconds=stride)
    window_size = datetime.timedelta(milliseconds=window_size)

    data_windows = {}

    for win_id in range(n_windows):
        start = origin + win_id * stride
        end = origin + win_id * stride + window_size

        for df_id, df in enumerate(df_list):
            df_sub = df.loc[(df.index > start) & (df.index < end), :].copy()
            df_sub["id"] = win_id
            if df_id not in data_windows:
                data_windows[df_id] = [df_sub]
            else:
                data_windows[df_id].append(df_sub)

    concat_df = [pd.concat(df, ignore_index=True) for df in data_windows.values()]
    final_features = []
    for df in concat_df:
        features = tsfresh.extract_features(
            timeseries_container=df,
            column_id="id",
            # column_sort="timestamp",
            default_fc_parameters=settings,
        )
        features = impute(features)  # Replace Nan and inf by with extreme values (min, max)
        final_features.append(features)

    features = pd.concat(final_features, axis=1)
    return features


def calculate_window_parameters(
        length: float,
        window_size: float,
        overlap: float,
) -> Tuple[int, int]:
    stride = int(window_size - (window_size * overlap))
    n_windows = math.floor((length - window_size) / stride) + 1
    return n_windows, stride
