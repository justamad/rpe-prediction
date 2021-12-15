from .statistical_features import calculate_statistical_features_with_sliding_window_time_based
from typing import Tuple

from src.config import (
    SubjectDataIterator,
    RPESubjectLoader,
    AzureDataFrameLoader,
    HeartRateDataFrameLoader,
    ImuDataFrameLoader,
)

import numpy as np
import pandas as pd


def calculate_all_features(
        input_path: str,
        window_size: int,
        overlap: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    iterator = SubjectDataIterator(
        base_path=input_path,
        loaders=[RPESubjectLoader, AzureDataFrameLoader, HeartRateDataFrameLoader, ImuDataFrameLoader],
    )

    x_data = []
    y_data = []

    for set_data in iterator.iterate_over_all_subjects():
        X = calculate_statistical_features_with_sliding_window_time_based(
            [set_data["azure"], set_data["ecg"], set_data["imu"]],
            window_size=window_size,
            overlap=overlap,
        )

        y_values = [set_data['subject_name'], set_data['rpe'], set_data['group'], set_data['nr_set']]
        y = pd.DataFrame(
            data=[y_values for _ in range(len(X))],
            columns=['name', 'rpe', 'group', 'set'],
        )

        x_data.append(X)
        y_data.append(y)

    return pd.concat(x_data, ignore_index=True), pd.concat(y_data, ignore_index=True)
