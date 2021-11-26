from typing import Tuple

from src.config import (
    SubjectDataIterator,
    RPESubjectLoader,
    AzureDataFrameLoader,
    ImuDataFrameLoader,
    HeartRateDataFrameLoader,
)

from src.processing import (
    remove_columns_from_dataframe,
)

import pandas as pd


def collect_all_trials_with_labels(input_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    file_iterator = SubjectDataIterator(
        base_path=input_path,
        loaders=[RPESubjectLoader, AzureDataFrameLoader, ImuDataFrameLoader, HeartRateDataFrameLoader],
    )

    x_data = []
    y_data = []

    for trial in file_iterator.iterate_over_all_subjects():
        X_df = pd.read_csv(trial['azure'], sep=';').set_index('timestamp', drop=True)
        imu_df = pd.read_csv(trial['imu'], sep=';').set_index('sensorTimestamp', drop=True)
        hr_df = pd.read_csv(trial['ecg'], sep=';').set_index('timestamp', drop=True)
        X_df = remove_columns_from_dataframe(X_df, ["FOOT"])

        y_values = [trial['subject_name'], trial['rpe'], trial['group'], trial['nr_set']]
        y = pd.DataFrame(
            data=[y_values for _ in range(len(X_df))],
            columns=['name', 'rpe', 'group', 'set', ]
        )

        x_data.append(X_df)
        y_data.append(y)

    return pd.concat(x_data, ignore_index=True), pd.concat(y_data, ignore_index=True)
