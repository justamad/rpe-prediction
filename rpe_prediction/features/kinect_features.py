from .sliding_window import calculate_features_sliding_window
from rpe_prediction.config import SubjectDataIterator, RPESubjectLoader, FusedAzureSubjectLoader

from rpe_prediction.processing import (
    create_rotation_matrix_y_axis,
    apply_affine_transformation,
    remove_columns_from_dataframe
)

from .skeleton_features import (
    calculate_3d_joint_velocities,
    calculate_joint_angles_with_reference_joint,
    calculate_angles_between_3_joints,
    calculate_1d_joint_velocities,
    calculate_relative_coordinates_with_reference_joint
)

import pandas as pd
import numpy as np

ROTATION_ANGLE = 15


def calculate_kinect_feature_set(input_path: str, window_size: int = 30, overlap: float = 0.5,
                                 data_augmentation: bool = False, nr_iterations: int = 10):
    file_iterator = SubjectDataIterator(input_path).add_loader(RPESubjectLoader).add_loader(FusedAzureSubjectLoader)
    x_data = []
    y_data = []

    for set_data in file_iterator.iterate_over_all_subjects():
        kinect_df = pd.read_csv(set_data['azure'], sep=';', index_col=False).set_index('timestamp', drop=True)
        kinect_df = remove_columns_from_dataframe(kinect_df, ["FOOT"])

        if data_augmentation:
            for _ in range(nr_iterations):
                angle = (np.random.rand() * 2 * ROTATION_ANGLE) - ROTATION_ANGLE
                matrix = create_rotation_matrix_y_axis(angle)
                df = apply_affine_transformation(kinect_df, matrix)
                x, y = extract_kinect_features(df, window_size, overlap, set_data, augmented=True)
                x_data.append(x)
                y_data.append(y)

        x, y = extract_kinect_features(kinect_df, window_size, overlap, set_data, augmented=False)
        x_data.append(x)
        y_data.append(y)

    return pd.concat(x_data, ignore_index=True), pd.concat(y_data, ignore_index=True)


def extract_kinect_features(df: pd.DataFrame, window_size: int, overlap: float, trial: dict, augmented: bool = False):
    velocities_1d = calculate_1d_joint_velocities(df)
    velocities_3d = calculate_3d_joint_velocities(df)
    relative_coordinates = calculate_relative_coordinates_with_reference_joint(df, "PELVIS")
    angle_three = calculate_angles_between_3_joints(df)
    angle_origin = calculate_joint_angles_with_reference_joint(df)
    angle = pd.concat([angle_three, angle_origin], axis=1)

    angles_velocity = angle.diff(axis=0).dropna(axis='index')
    angles_velocity.rename(lambda c: c + "_VELOCITY", axis='columns', inplace=True)

    features = pd.concat([velocities_1d,
                          velocities_3d,
                          relative_coordinates.iloc[1:],
                          angle.iloc[1:],
                          angles_velocity,
                          ], axis=1).reset_index()
    x = calculate_features_sliding_window(features, window_size=window_size, overlap=overlap)

    # Construct y-data with pseudonyms, rpe values, groups and number of sets
    y = np.repeat([[trial['subject_name'], trial['rpe'], trial['group'], trial['nr_set'], augmented]],
                  len(x), axis=0)
    y = pd.DataFrame(y, columns=['name', 'rpe', 'group', 'set', 'augmented'])
    return x, y
