from rpe_prediction.config import SubjectDataIterator, RPESubjectLoader, FusedAzureSubjectLoader

from rpe_prediction.processing import (
    create_rotation_matrix_y_axis,
    apply_affine_transformation,
    remove_columns_from_dataframe
)

import pandas as pd
import numpy as np


def prepare_data_for_deep_learning(
        input_path: str,
        nr_augmentation_iterations: int = 0,
):
    file_iterator = SubjectDataIterator(input_path).add_loader(RPESubjectLoader).add_loader(FusedAzureSubjectLoader)
    x_data = []
    y_data = []

    for trial in file_iterator.iterate_over_all_subjects():
        X_df = pd.read_csv(trial['azure'], sep=';', index_col=False).set_index('timestamp', drop=True)
        X_df = remove_columns_from_dataframe(X_df, ["FOOT"])

        # for _ in range(nr_augmentation_iterations):
        #     angle = (np.random.rand() * 2 * ROTATION_ANGLE) - ROTATION_ANGLE
        #     matrix = create_rotation_matrix_y_axis(angle)
        #     df = apply_affine_transformation(kinect_df, matrix)
        #     x, y = extract_kinect_features(df, window_size, overlap, set_data, augmented=True)
        #     x_data.append(x)
        #     y_data.append(y)

        y_values = [trial['subject_name'], trial['rpe'], trial['group'], trial['nr_set']]
        y = pd.DataFrame(data=[y_values for _ in range(len(X_df))],
                         columns=['name', 'rpe', 'group', 'set', ])

        x_data.append(X_df)
        y_data.append(y)

    return pd.concat(x_data, ignore_index=True), pd.concat(y_data, ignore_index=True)
