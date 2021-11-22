import numpy as np
import pandas as pd


def create_rotation_matrix_y_axis(angle_degrees: float):
    angle_rad = angle_degrees * np.pi / 180
    return np.matrix([[np.cos(angle_rad), 0, np.sin(angle_rad)],
                      [0, 1, 0],
                      [-np.sin(angle_rad), 0, np.cos(angle_rad)]])


def create_rotation_matrix_z_axis(angle_degrees: float):
    angle_rad = angle_degrees * np.pi / 180
    return np.matrix([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                      [np.sin(angle_rad), np.cos(angle_rad), 0],
                      [0, 0, 1]])


def calculate_angle_in_radians_between_vectors(v1: np.ndarray, v2: np.ndarray):
    l_v1 = v1 / np.linalg.norm(v1, axis=1).reshape(-1, 1)
    l_v2 = v2 / np.linalg.norm(v2, axis=1).reshape(-1, 1)
    dot = np.sum(l_v1 * l_v2, axis=1)
    return np.arccos(dot) * 180 / np.pi


def apply_affine_transformation(
        df: pd.DataFrame,
        matrix: np.ndarray,
        translation: np.ndarray = np.array([0, 0, 0])
) -> pd.DataFrame:
    samples, features = df.shape
    result = matrix * df.to_numpy().reshape(-1, 3).T + translation.reshape(3, 1)
    final_result = result.T.reshape(samples, features)
    return pd.DataFrame(data=final_result, columns=df.columns, index=df.index)
