from .statistical_features import calculate_statistical_features_with_sliding_window
from typing import Tuple

from src.dataset import (
    SubjectDataIterator,
    RPESubjectLoader,
)

from src.processing import (
    create_rotation_matrix_y_axis,
    apply_affine_transformation,
    remove_columns_from_dataframe,
)

from .skeleton_features import (
    calculate_3d_joint_velocities,
    calculate_joint_angles_with_reference_joint,
    calculate_angles_between_3_joints,
    calculate_1d_joint_velocities,
    calculate_relative_coordinates_with_reference_joint,
)

import pandas as pd
import numpy as np

ROTATION_ANGLE = 7.5


def calculate_kinect_feature_set(
        input_path: str,
        statistical_features: bool = True,
        window_size: int = 30,
        overlap: float = 0.5,
        nr_augmentation_iterations: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    file_iterator = SubjectDataIterator(
        base_path=input_path,
        log_path=input_path,
        loaders=[RPESubjectLoader, AzureDataFrameLoader],
    )

    x_data = []
    y_data = []

    for set_data in file_iterator.iterate_over_all_subjects():
        kinect_df = pd.read_csv(set_data['azure'], sep=';', index_col=False).set_index('timestamp', drop=True)
        kinect_df = remove_columns_from_dataframe(kinect_df, ["FOOT"])

        for _ in range(nr_augmentation_iterations):
            angle = (np.random.rand() * 2 * ROTATION_ANGLE) - ROTATION_ANGLE
            matrix = create_rotation_matrix_y_axis(angle)
            df = apply_affine_transformation(kinect_df, matrix)
            x, y = extract_kinect_features(df, statistical_features, window_size, overlap, set_data, augmented=True)
            x_data.append(x)
            y_data.append(y)

        x, y = extract_kinect_features(kinect_df, statistical_features, window_size, overlap, set_data, augmented=False)
        x_data.append(x)
        y_data.append(y)

    return pd.concat(x_data, ignore_index=True), pd.concat(y_data, ignore_index=True)


def extract_kinect_features(
        df: pd.DataFrame,
        statistical_features: bool,
        window_size: int,
        overlap: float,
        trial: dict,
        augmented: bool = False,
):
    velocities_1d = calculate_1d_joint_velocities(df)
    velocities_3d = calculate_3d_joint_velocities(df)
    relative_coordinates = calculate_relative_coordinates_with_reference_joint(df, "PELVIS")
    angle_three = calculate_angles_between_3_joints(df)
    angle_origin = calculate_joint_angles_with_reference_joint(df)
    angle = pd.concat([angle_three, angle_origin], axis=1)

    angles_velocity = angle.diff(axis=0).dropna(axis='index')
    angles_velocity.rename(lambda c: c + "_VELOCITY", axis='columns', inplace=True)

    X = pd.concat([velocities_1d,
                   velocities_3d,
                   relative_coordinates.iloc[1:],
                   angle.iloc[1:],
                   angles_velocity,
                   ], axis=1).reset_index()

    if statistical_features:
        X = calculate_statistical_features_with_sliding_window(
            df=X,
            window_size=window_size,
            overlap=overlap
        )

    y_values = [trial['subject_name'], trial['rpe'], trial['group'], trial['nr_set'], augmented]
    y = pd.DataFrame(
        data=[y_values for _ in range(len(X))],
        columns=['name', 'rpe', 'group', 'set', 'augmented']
    )

    return X, y
