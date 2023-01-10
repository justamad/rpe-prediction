import numpy as np
import pandas as pd
import math


# def create_euler_rot_xyz_from_degrees(x: float, y: float, z: float):
#     rot = create_rotation_matrix_x(x)
#     rot = rot * create_rotation_matrix_y_axis(y)
#     rot = rot * create_rotation_matrix_z_axis(z)
#     return rot


def create_euler_rotation_matrix(x: float, y: float, z: float):
    x_rot = np.matrix([
        [1, 0, 0],
        [0, math.cos(x), -math.sin(x)],
        [0, math.sin(x), math.cos(x)]
    ], np.float)

    y_rot = np.matrix([
        [math.cos(y), 0, math.sin(y)],
        [0, 1, 0],
        [-math.sin(y), 0, math.cos(y)]
    ], np.float)

    z_rot = np.matrix([
        [math.cos(z), -math.sin(z), 0],
        [math.sin(z), math.cos(z), 0],
        [0, 0, 1]
    ], np.float)

    return z_rot * y_rot * x_rot


# def create_rotation_matrix_x(angle_deg: float):
#     angle_deg = angle_deg * np.pi / 180
#     return np.matrix([[1, 0, 0],
#                       [0, np.cos(angle_deg), -np.sin(angle_deg)],
#                       [0, np.sin(angle_deg), np.cos(angle_deg)]])
#
#
# def create_rotation_matrix_y_axis(angle_degrees: float):
#     angle_rad = angle_degrees * np.pi / 180
#     return np.matrix([[np.cos(angle_rad), 0, np.sin(angle_rad)],
#                       [0, 1, 0],
#                       [-np.sin(angle_rad), 0, np.cos(angle_rad)]])
#
#
# def create_rotation_matrix_z_axis(angle_degrees: float):
#     angle_rad = angle_degrees * np.pi / 180
#     return np.matrix([[np.cos(angle_rad), -np.sin(angle_rad), 0],
#                       [np.sin(angle_rad), np.cos(angle_rad), 0],
#                       [0, 0, 1]])


def create_unit_vectors(v: np.ndarray) -> np.ndarray:
    unit_vector = v / np.linalg.norm(v, axis=1).reshape(-1, 1)
    return unit_vector


def calculate_angle_between_vectors_deg(
        v1: np.ndarray,
        v2: np.ndarray,
) -> np.ndarray:
    v1_u = create_unit_vectors(v1)
    v2_u = create_unit_vectors(v2)
    product = np.sum(v1_u * v2_u, axis=1)
    clipped = np.clip(product, -1.0, 1.0)
    return np.arccos(clipped) * 180 / np.pi


def apply_affine_transformation_to_df(
        df: pd.DataFrame,
        matrix: np.ndarray,
        translation: np.ndarray = np.array([0, 0, 0]),
) -> pd.DataFrame:
    samples, features = df.shape
    result = matrix * df.to_numpy().reshape(-1, 3).T + translation.reshape(3, 1)
    final_result = result.T.reshape(samples, features)
    return pd.DataFrame(data=final_result, columns=df.columns, index=df.index)
