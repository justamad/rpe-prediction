from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

import tsfresh
import pandas as pd
import math

settings = EfficientFCParameters()
del settings['variance']  # Variance and standard deviation are highly correlated but std integrates nr of samples
del settings['length']  # Length is constant for all windows
del settings['sum_values']  # Highly correlated with RMS and Mean
del settings['mean']  # Highly correlated with RMS and Sum
del settings['variance_larger_than_standard_deviation']
del settings['has_duplicate_max']
del settings['has_duplicate_min']
del settings['has_duplicate']
settings = {k: v for k, v in settings.items() if not isinstance(v, list)}


def calculate_statistical_features_with_sliding_window(
        df: pd.DataFrame,
        window_size: int,
        overlap: float = 0.25
):
    windows = []
    stride = int(window_size - (window_size * overlap))
    n_windows = math.floor((len(df) - window_size) / stride) + 1

    for window_idx in range(0, n_windows):
        window = df.iloc[window_idx * stride:window_idx * stride + window_size].copy()
        window['id'] = window_idx
        windows.append(window)

    df = pd.concat(windows, ignore_index=True)
    features = tsfresh.extract_features(
        timeseries_container=df,
        column_id='id',
        column_sort='timestamp',
        default_fc_parameters=settings,
    )

    features = impute(features)  # Replace Nan and inf by with extreme values (min, max)
    return features