from scipy.spatial.transform import Rotation as R
from typing import Dict
from enum import Enum

import pandas as pd
import numpy as np
import math


boundaries = np.array([
    (-90, 90),  # 00, PELVIS (x)
    (-90, 90),  # 01, PELVIS (y)
    (-90, 90),  # 02, PELVIS (z)
    (-90, 90),  # 03, HIP_LEFT (x)
    (-90, 90),  # 04, HIP_LEFT (y)
    (-90, 90),  # 05, HIP_LEFT (z)
    (-120, 120),  # 06, KNEE_LEFT (x)
    (-90, 90),  # 07, ANKLE_LEFT (x)
    (-10, 10),  # 08, ANKLE_LEFT (y)
    (-90, 90),  # 09, HIP_RIGHT (x)
    (-90, 90),  # 10, HIP_RIGHT (y)
    (-90, 90),  # 11, HIP_RIGHT (z)
    (-120, 120),  # 12, KNEE_RIGHT (x)
    (-90, 90),  # 13, ANKLE_RIGHT (x)
    (-10, 10),  # 14, ANKLE_RIGHT (y)
    (-45, 45),  # 15, SPINE_NAVEL (x)
    (-45, 45),  # 16, SPINE_NAVEL (y)
    (-45, 45),  # 17, SPINE_NAVEL (z)
    (-45, 45),  # 18, SPINE_CHEST (x)
    (-45, 45),  # 19, SPINE_CHEST (y)
    (-45, 45),  # 20, SPINE_CHEST (z)
    (-45, 45),  # 21, NECK (x)
    (-45, 45),  # 22, NECK (y)
    (-45, 45),  # 23, NECK (z)
    (-90, 90),  # 24, SHOULDER_LEFT (y)
    (-90, 90),  # 25, SHOULDER_LEFT (z)
    (-90, 90),  # 26, ELBOW_LEFT (x)
    (-5, 5),  # 27, ELBOW_LEFT (y)
    (-5, 5),  # 28, ELBOW_LEFT (z)
    (-90, 90),  # 29, WRIST_LEFT (z)
    (-90, 90),  # 30, SHOULDER_RIGHT (y)
    (-90, 90),  # 31, SHOULDER_RIGHT (z)
    (-90, 90),  # 32, ELBOW_RIGHT (x)
    (-5, 5),  # 33, ELBOW_RIGHT (y)
    (-5, 5),  # 34, ELBOW_RIGHT (z)
    (-90, 90),  # 35, WRIST_RIGHT (z)
])


class Skeleton(Enum):
    PELVIS = 0
    HIP_LEFT = 1
    KNEE_LEFT = 2
    ANKLE_LEFT = 3
    FOOT_LEFT = 4
    HIP_RIGHT = 5
    KNEE_RIGHT = 6
    ANKLE_RIGHT = 7
    FOOT_RIGHT = 8
    SPINE_NAVEL = 9
    SPINE_CHEST = 10
    NECK = 11
    SHOULDER_LEFT = 12
    ELBOW_LEFT = 13
    WRIST_LEFT = 14
    SHOULDER_RIGHT = 15
    ELBOW_RIGHT = 16
    WRIST_RIGHT = 17


KINEMATIC_CHAIN = [
    (Skeleton.PELVIS, Skeleton.HIP_LEFT),
    (Skeleton.HIP_LEFT, Skeleton.KNEE_LEFT),
    (Skeleton.KNEE_LEFT, Skeleton.ANKLE_LEFT),
    (Skeleton.ANKLE_LEFT, Skeleton.FOOT_LEFT),
    (Skeleton.PELVIS, Skeleton.HIP_RIGHT),
    (Skeleton.HIP_RIGHT, Skeleton.KNEE_RIGHT),
    (Skeleton.KNEE_RIGHT, Skeleton.ANKLE_RIGHT),
    (Skeleton.ANKLE_RIGHT, Skeleton.FOOT_RIGHT),
    (Skeleton.PELVIS, Skeleton.SPINE_NAVEL),
    (Skeleton.SPINE_NAVEL, Skeleton.SPINE_CHEST),
    (Skeleton.SPINE_CHEST, Skeleton.NECK),
    (Skeleton.NECK, Skeleton.SHOULDER_LEFT),
    (Skeleton.SHOULDER_LEFT, Skeleton.ELBOW_LEFT),
    (Skeleton.ELBOW_LEFT, Skeleton.WRIST_LEFT),
    (Skeleton.NECK, Skeleton.SHOULDER_RIGHT),
    (Skeleton.SHOULDER_RIGHT, Skeleton.ELBOW_RIGHT),
    (Skeleton.ELBOW_RIGHT, Skeleton.WRIST_RIGHT),
]


def calculate_forwards_kinematics(angles: np.array, bone_lengths: Dict) -> np.array:
    skeleton = np.zeros((len(Skeleton), 3))
    skeleton[0, :] = np.zeros(3)

    # PELVIS
    hip_rot = create_euler_rotation_matrix(angles[0], angles[1], angles[2]) * create_euler_rotation_matrix(
        180 * math.pi / 180, 0, 0)

    # HIP_LEFT
    skeleton[1, :] = bone_lengths["PELVIS-HIP_LEFT"] * np.array([1, 0, 0]) * hip_rot + skeleton[0, :]

    # KNEE_LEFT
    left_leg_rot = create_euler_rotation_matrix(angles[3], angles[4], angles[5]) * hip_rot
    skeleton[2, :] = bone_lengths["HIP_LEFT-KNEE_LEFT"] * np.array([0, -1, 0]) * left_leg_rot + skeleton[1, :]

    # ANKLE_LEFT
    left_knee_rot = create_euler_rotation_matrix(angles[6], 0, 0) * left_leg_rot
    skeleton[3, :] = bone_lengths["KNEE_LEFT-ANKLE_LEFT"] * np.array([0, -1, 0]) * left_knee_rot + skeleton[2, :]

    # FOOT_LEFT
    left_ankle_rot = create_euler_rotation_matrix(angles[7], angles[8], 0) * left_knee_rot
    skeleton[4, :] = bone_lengths["ANKLE_LEFT-FOOT_LEFT"] * np.array([0, 0, 1]) * left_ankle_rot + skeleton[3, :]

    # HIP_RIGHT
    skeleton[5, :] = bone_lengths["PELVIS-HIP_RIGHT"] * np.array([-1, 0, 0]) * hip_rot + skeleton[0, :]

    # KNEE_RIGHT
    right_leg_rot = create_euler_rotation_matrix(angles[9], angles[10], angles[11]) * hip_rot
    skeleton[6, :] = bone_lengths["HIP_RIGHT-KNEE_RIGHT"] * np.array([0, -1, 0]) * right_leg_rot + skeleton[5, :]

    # ANKLE_RIGHT
    right_knee_rot = create_euler_rotation_matrix(angles[12], 0, 0) * right_leg_rot
    skeleton[7, :] = bone_lengths["KNEE_RIGHT-ANKLE_RIGHT"] * np.array([0, -1, 0]) * right_knee_rot + skeleton[6, :]

    # FOOT_RIGHT
    right_ankle_rot = create_euler_rotation_matrix(angles[13], angles[14], 0) * right_knee_rot
    skeleton[8, :] = bone_lengths["ANKLE_RIGHT-FOOT_RIGHT"] * np.array([0, 0, 1]) * right_ankle_rot + skeleton[7, :]

    # PELVIS to SPINE_NAVEL
    spine_rot = create_euler_rotation_matrix(angles[15], angles[16], angles[17]) * hip_rot
    skeleton[9, :] = bone_lengths["PELVIS-SPINE_NAVEL"] * np.array([0, 1, 0]) * spine_rot + skeleton[0, :]

    # SPINE_NAVEL to SPINE_CEST
    spine_chest_rot = create_euler_rotation_matrix(angles[18], angles[19], angles[20]) * spine_rot
    skeleton[10, :] = bone_lengths["SPINE_NAVEL-SPINE_CHEST"] * np.array([0, 1, 0]) * spine_chest_rot + skeleton[9, :]

    # SPINE_CHEST to NECK
    neck_rot = create_euler_rotation_matrix(angles[21], angles[22], angles[23]) * spine_chest_rot
    skeleton[11, :] = bone_lengths["SPINE_CHEST-NECK"] * np.array([0, 1, 0]) * neck_rot + skeleton[10, :]

    # NECK to SHOULDER_LEFT
    shoulder_left_rot = create_euler_rotation_matrix(0, angles[24], angles[25]) * neck_rot
    skeleton[12, :] = bone_lengths["NECK-SHOULDER_LEFT"] * np.array([1, 0, 0]) * shoulder_left_rot + skeleton[11, :]

    # SHOULDER_LEFT to ELBOW_LEFT
    elbow_left_rot = create_euler_rotation_matrix(angles[26], angles[27], angles[28]) * shoulder_left_rot
    skeleton[13, :] = bone_lengths["SHOULDER_LEFT-ELBOW_LEFT"] * np.array([1, 0, 0]) * elbow_left_rot + skeleton[12, :]

    # ELBOW_LEFT to WRIST_LEFT
    wrist_left_rot = create_euler_rotation_matrix(0, 0, angles[29]) * elbow_left_rot
    skeleton[14, :] = bone_lengths["ELBOW_LEFT-WRIST_LEFT"] * np.array([1, 0, 0]) * wrist_left_rot + skeleton[13, :]

    # NECK to SHOULDER_RIGHT
    shoulder_right_rot = create_euler_rotation_matrix(0, angles[30], angles[31]) * neck_rot
    skeleton[15, :] = bone_lengths["NECK-SHOULDER_RIGHT"] * np.array([-1, 0, 0]) * shoulder_right_rot + skeleton[11, :]

    # SHOULDER_RIGHT to ELBOW_RIGHT
    elbow_right_rot = create_euler_rotation_matrix(angles[32], angles[33], angles[34]) * shoulder_right_rot
    skeleton[16, :] = bone_lengths["SHOULDER_RIGHT-ELBOW_RIGHT"] * np.array([-1, 0, 0]) * elbow_right_rot + skeleton[15, :]

    # ELBOW_RIGHT to WRIST_RIGHT
    wrist_right_rot = create_euler_rotation_matrix(0, 0, angles[35]) * elbow_right_rot
    skeleton[17, :] = bone_lengths["ELBOW_RIGHT-WRIST_RIGHT"] * np.array([-1, 0, 0]) * wrist_right_rot + skeleton[16, :]
    return skeleton


def complete_restricted_angle_vector_with_zeros(angles: np.ndarray) -> np.ndarray:
    angles = list(angles)
    angles.insert(7, 0)  # KNEE_LEFT (y)
    angles.insert(8, 0)  # KNEE_LEFT (z)
    angles.insert(11, 0)  # ANKLE_LEFT (z)

    # FOOT_LEFT (x,y,z)
    angles.insert(12, 0)
    angles.insert(13, 0)
    angles.insert(14, 0)

    angles.insert(19, 0)  # KNEE_RIGHT (y)
    angles.insert(20, 0)  # KNEE_RIGHT (z)
    angles.insert(23, 0)  # ANKLE_RIGHT (z)

    # FOOT_RIGHT (x,y,z)
    angles.insert(24, 0)
    angles.insert(25, 0)
    angles.insert(26, 0)

    angles.insert(36, 0)  # SHOULDER_LEFT (x)
    angles.insert(42, 0)  # WRIST_LEFT (x)
    angles.insert(43, 0)  # WRIST_LEFT (y)
    angles.insert(45, 0)  # SHOULDER_RIGHT (x)
    angles.insert(51, 0)  # WRIST_RIGHT (x)
    angles.insert(52, 0)  # WRIST_RIGHT (y)
    return np.array(angles)


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


def multiply_kinematic_chain_entire_trial(trial: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        data=[create_euler_kinematic_chain(trial.iloc[i, :]) for i in range(trial.shape[0])],
        columns=trial.columns
    )
    return df


def create_euler_kinematic_chain(angles: np.ndarray):
    joint_rotations = [R.from_euler("xyz", angles[0:3]).as_matrix()]  # Start with original hip ori

    for last_joint, cur_joint in KINEMATIC_CHAIN:
        rotation = R.from_euler("xyz", angles[cur_joint * 3: (cur_joint + 1) * 3]).as_matrix()
        last_rotation = joint_rotations[last_joint]
        cur_rotation = np.matmul(last_rotation, rotation)
        joint_rotations.append(cur_rotation)

    euler_angles = [R.from_matrix(r).as_euler("xyz") for r in joint_rotations]  # Convert rot matrices to euler angles
    return np.array(euler_angles).flatten()


def calculate_bone_segments(df: pd.DataFrame) -> Dict[str, float]:
    bone_lengths = {}

    for j1, j2 in KINEMATIC_CHAIN:
        j1 = j1.name
        j2 = j2.name
        marker_1 = df[[f"{j1} ({c})" for c in ["x", "y", "z"]]].values
        marker_2 = df[[f"{j2} ({c})" for c in ["x", "y", "z"]]].values

        lengths = np.linalg.norm(marker_1 - marker_2, axis=1)
        mean, std = lengths.mean(), lengths.std()
        bone_lengths[f"{j1}-{j2}"] = mean

    return bone_lengths


def restructure_data_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = [f"{c.name} ({d})" for c in list(Skeleton) for d in ["x", "y", "z"]]
    df_new = df[columns]
    return df_new
