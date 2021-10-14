from rpe_prediction.config import SubjectDataIterator, RPESubjectLoader, FusedAzureSubjectLoader

from rpe_prediction.processing import (
    remove_columns_from_dataframe
)

import pandas as pd


def collect_all_trials_with_labels(input_path: str):
    file_iterator = SubjectDataIterator(input_path).add_loader(RPESubjectLoader).add_loader(FusedAzureSubjectLoader)
    x_data = []
    y_data = []

    for trial in file_iterator.iterate_over_all_subjects():
        X_df = pd.read_csv(trial['azure'], sep=';', index_col=False).set_index('timestamp', drop=True)
        X_df = remove_columns_from_dataframe(X_df, ["FOOT"])

        y_values = [trial['subject_name'], trial['rpe'], trial['group'], trial['nr_set']]
        y = pd.DataFrame(data=[y_values for _ in range(len(X_df))],
                         columns=['name', 'rpe', 'group', 'set', ])

        x_data.append(X_df)
        y_data.append(y)

    return pd.concat(x_data, ignore_index=True), pd.concat(y_data, ignore_index=True)


def collect_all_trials_with_labels_own_generator(input_path: str):
    file_iterator = SubjectDataIterator(input_path).add_loader(RPESubjectLoader).add_loader(FusedAzureSubjectLoader)
    x_data = []
    y_data = []

    for trial in file_iterator.iterate_over_all_subjects():
        X_df = pd.read_csv(trial['azure'], sep=';', index_col=False).set_index('timestamp', drop=True)
        X_df = remove_columns_from_dataframe(X_df, ["FOOT"])

        x_data.append(X_df)
        y_data.append((trial['subject_name'], trial['rpe'], trial['group'], trial['nr_set']))

    return x_data, y_data
