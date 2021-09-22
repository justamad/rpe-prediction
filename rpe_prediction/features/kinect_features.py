from .sliding_window import calculate_features_sliding_window
from rpe_prediction.config import SubjectDataIterator, RPESubjectLoader, FusedAzureSubjectLoader
from rpe_prediction.processing import create_rotation_matrix_y_axis, apply_affine_transformation

from .skeleton_features import (
    calculate_3d_joint_velocities,
    calculate_joint_angles_with_reference_joint,
    calculate_angles_between_3_joints,
    calculate_individual_axes_joint_velocities)

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

        if data_augmentation:
            for _ in range(nr_iterations):
                angle = np.random.randint(-ROTATION_ANGLE, ROTATION_ANGLE)
                matrix = create_rotation_matrix_y_axis(angle)
                df = apply_affine_transformation(kinect_df, matrix)
                x, y = extract_kinect_features(df, window_size, overlap, set_data)
                x_data.append(x)
                y_data.append(y)

        x, y = extract_kinect_features(kinect_df, window_size, overlap, set_data)
        x_data.append(x)
        y_data.append(y)

    return pd.concat(x_data, ignore_index=True), pd.concat(y_data, ignore_index=True)


def extract_kinect_features(df: pd.DataFrame, window_size: int, overlap: float, set_data: dict):
    velocities = calculate_individual_axes_joint_velocities(df)
    velocity_3d = calculate_3d_joint_velocities(df)
    angle_three = calculate_angles_between_3_joints(df)
    angle_origin = calculate_joint_angles_with_reference_joint(df)
    angle = pd.concat([angle_three, angle_origin], axis=1)

    angles_velocity = angle.diff(axis=0).dropna(axis='index')
    angles_velocity.rename(lambda c: c + "_SPEED", axis='columns', inplace=True)

    features = pd.concat([velocities, velocity_3d, angle.iloc[1:], angles_velocity], axis=1).reset_index()
    x = calculate_features_sliding_window(features, window_size=window_size, overlap=overlap)

    # Construct y-data with pseudonyms, rpe values, groups and number of sets
    y = np.repeat([[set_data['subject_name'], set_data['rpe'], set_data['group'], set_data['nr_set']]],
                  len(features), axis=0)
    y = pd.DataFrame(y, columns=['name', 'rpe', 'group', 'set'])
    return x, y
