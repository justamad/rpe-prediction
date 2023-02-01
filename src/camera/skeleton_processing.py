from src.processing import apply_butterworth_filter, apply_affine_transformation
from src.camera.kabsch import find_rigid_transformation_svd
from typing import List
from .fitting import fit_inverse_kinematic_parallel

import numpy as np
import pandas as pd


excluded_joints = ["EYE", "EAR", "NOSE", "HANDTIP", "THUMB", "CLAVICLE", "HAND"]


def fuse_cameras(df1: pd.DataFrame, df2: pd.DataFrame):
    df1 = process_data_frame_initially(df1)
    df2 = process_data_frame_initially(df2)
    df1, df2 = synchronize_skeleton_on_timestamp(df1, df2)
    df1, df2 = calculate_affine_transformation_based_on_data(df1, df2)

    averaged = fuse_skeleton_gradients(df1, df2, exp=1.8, initial_poses=5)
    average_filtered = apply_butterworth_filter(
        df=averaged,
        cutoff=4,
        sampling_rate=30,
        order=4,
    )

    average_filtered = average_filtered.set_index(df1.index)
    fit_pos, fit_ori = fit_inverse_kinematic_parallel(average_filtered)
    fit_pos.index = pd.to_datetime(average_filtered.index, unit="s")
    fit_ori.index = pd.to_datetime(average_filtered.index, unit="s")
    return fit_pos, fit_ori


def process_data_frame_initially(df: pd.DataFrame) -> pd.DataFrame:
    df = select_main_person(df)
    df = df[[c for c in df.columns if "(c)" not in c]]
    df = remove_columns_from_dataframe(df)
    df.index *= 1e-6
    return df


def remove_columns_from_dataframe(df: pd.DataFrame):
    for excluded_part in excluded_joints:
        df = df.loc[:, ~df.columns.str.contains(excluded_part)]
    return df


def select_main_person(df: pd.DataFrame) -> pd.DataFrame:
    body_idx_c = df["body_idx"].value_counts()
    df = df[df["body_idx"] == body_idx_c.index[body_idx_c.argmax()]]
    df = df.drop("body_idx", axis=1)
    return df


def synchronize_skeleton_on_timestamp(*data_frames) -> List[pd.DataFrame]:
    start_time = max([df.index[0] for df in data_frames])
    end_time = min([df.index[-1] for df in data_frames])

    result = [df[(df.index >= start_time) & (df.index < end_time)] for df in data_frames]
    max_len = min([len(df) for df in result])
    result = [df.iloc[:max_len] for df in result]
    return result


def fuse_skeleton_gradients(df1: pd.DataFrame, df2: pd.DataFrame, exp: float = 1.4, initial_poses: int = 20):
    x1 = df1.values
    x2 = df2.values

    result = np.zeros(x1.shape)
    result[:initial_poses, :] = (x1[:initial_poses, :] + x2[:initial_poses, :]) / 2
    for frame in range(initial_poses, len(x1)):
        for joint in range(x1.shape[1] // 3):
            last_point = result[frame - 1, joint * 3:joint * 3 + 3]

            p1 = x1[frame, joint * 3:joint * 3 + 3]
            p2 = x2[frame, joint * 3:joint * 3 + 3]

            # Distance with respect to last averaged point
            dist1 = np.linalg.norm(p1 - last_point) ** exp
            dist2 = np.linalg.norm(p2 - last_point) ** exp

            dist1 = 1e-3 if dist1 < 1e-5 else dist1
            dist2 = 1e-3 if dist2 < 1e-5 else dist2

            w1 = 1.0 / dist1
            w2 = 1.0 / dist2

            weight_sum = w1 + w2
            w1 /= weight_sum
            w2 /= weight_sum

            result[frame, joint * 3:joint * 3 + 3] = w1 * p1 + w2 * p2

    return pd.DataFrame(result, columns=df1.columns)


def calculate_affine_transformation_based_on_data(df1: pd.DataFrame, df2: pd.DataFrame):
    rotation, translation = find_rigid_transformation_svd(
        df1.values.reshape(-1, 3),
        df2.values.reshape(-1, 3),
        show=False,
    )
    df1 = apply_affine_transformation(df1, rotation, translation)
    return df1, df2


def calculate_error_between_both_cameras(df1: pd.DataFrame, df2: pd.DataFrame):
    diff = (df1.values - df2.values) ** 2
    distances = np.sqrt(diff.reshape(-1, 3).sum(axis=1))
    return np.mean(distances), np.std(distances)
