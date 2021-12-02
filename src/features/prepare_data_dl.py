from .normalization import normalize_skeleton_positions
from typing import Tuple, List

from src.config import (
    SubjectDataIterator,
    RPESubjectLoader,
    AzureDataFrameLoader,
    ImuDataFrameLoader,
    HeartRateDataFrameLoader,
)

from src.processing import (
    remove_columns_from_dataframe,
    apply_affine_transformation,
    align_skeleton_parallel_to_x_axis,
)

import pandas as pd
import numpy as np


def find_rotation_matrix(a, b):
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r = np.eye(3) + vx + np.dot(vx, vx) * (1 - c) / (s ** 2)
    return np.matrix(r)


def test_vector(R, a):
    return np.matmul(R, a)


a = np.array([0.030, 0.993, 0.122], dtype=np.float64)
b = np.array([0, 1, 0], dtype=np.float64)
R = find_rotation_matrix(a, b)
t = np.array([0, -1214, 0])


def collect_all_trials_with_labels(
        input_path: str,
        # show: bool = False,
) -> Tuple[List, pd.DataFrame]:
    file_iterator = SubjectDataIterator(
        base_path=input_path,
        loaders=[RPESubjectLoader, AzureDataFrameLoader, ImuDataFrameLoader, HeartRateDataFrameLoader],
    )

    x_data = []
    y_data = []

    for trial in file_iterator.iterate_over_all_subjects():
        skeleton_df = pd.read_csv(trial['azure'], sep=';').set_index('timestamp', drop=True)

        skeleton_df = apply_affine_transformation(skeleton_df, R, t)
        skeleton_df = normalize_skeleton_positions(skeleton_df, 'PELVIS')
        skeleton_df = align_skeleton_parallel_to_x_axis(skeleton_df)
        skeleton_df = remove_columns_from_dataframe(skeleton_df, ["PELVIS"])
        skeleton_df = (skeleton_df - skeleton_df.mean()) / skeleton_df.std()

        # if show:
        #     skeleton_df = normalize_skeleton_positions(skeleton_df, 'PELVIS')
        #     skeleton_df = align_skeleton_parallel_to_x_axis(skeleton_df)
        #     window = MoCapViewer(grid_axis='xz', sampling_frequency=30)
        #     window.add_skeleton(skeleton_df, skeleton_connection='azure', color='red')
        #     window.add_skeleton(skeleton_df_norm, skeleton_connection='azure', color='gray')
        #     window.show_window()

        imu_df = pd.read_csv(trial['imu'], sep=';').set_index('sensorTimestamp', drop=True)
        imu_df = (imu_df - imu_df.mean()) / imu_df.std()
        # hr_df = pd.read_csv(trial['ecg'], sep=';').set_index('timestamp', drop=True)

        y_values = [trial['subject_name'], trial['rpe'], trial['group'], trial['nr_set']]
        x_data.append((skeleton_df, imu_df))
        y_data.append(y_values)

    y_data = pd.DataFrame(data=y_data, columns=['name', 'rpe', 'group', 'nr_set'])
    return x_data, y_data
