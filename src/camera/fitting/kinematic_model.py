from .matrices import create_euler_rotation_matrix
from scipy.spatial.transform import Rotation as R
from typing import Dict

from PyMoCapViewer import MoCapViewer

import numpy as np
import math

skeleton_connection = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10),
]

""" Joint order in kinematic fit
00: PELVIS
01: HIP_LEFT
02: KNEE_LEFT
03: ANKLE_LEFT
04: FOOT_LEFT
05: HIP_RIGHT
06: KNEE_RIGHT
07: ANKLE_RIGHT
08: FOOT_RIGHT
09: SPINE_NAVEL
10: SPINE_CHEST
# 12: Nose (reserved)
# 13: Shoulder left
# 14: left arm
# 15: left wrist
# 16: left finger (TODO)
# 17: Shoulder right
# 18: right arm
# 19: right wrist
# 20: right finger (TODO)
"""


def build_skeleton(angles: np.array, bone_lengths: Dict) -> np.array:
    skeleton = np.zeros((11, 3))
    skeleton[0, :] = np.zeros(3)

    # PELVIS
    hip_rot = create_euler_rotation_matrix(angles[0], angles[1], angles[2]) * create_euler_rotation_matrix(180 * math.pi / 180, 0, 0)

    # HIP_LEFT
    skeleton[1, :] = bone_lengths["HIP_LEFT-PELVIS"] * np.array([1, 0, 0]) * hip_rot + skeleton[0, :]

    # KNEE_LEFT
    left_leg_rot = create_euler_rotation_matrix(angles[3], angles[4], angles[5]) * hip_rot
    skeleton[2, :] = bone_lengths["KNEE_LEFT-HIP_LEFT"] * np.array([0, -1, 0]) * left_leg_rot + skeleton[1, :]

    # ANKLE_LEFT
    left_knee_rot = create_euler_rotation_matrix(angles[6], 0, 0) * left_leg_rot
    skeleton[3, :] = bone_lengths["ANKLE_LEFT-KNEE_LEFT"] * np.array([0, -1, 0]) * left_knee_rot + skeleton[2, :]

    # FOOT_LEFT
    left_ankle_rot = create_euler_rotation_matrix(angles[7], angles[8], 0) * left_knee_rot
    skeleton[4, :] = bone_lengths["FOOT_LEFT-ANKLE_LEFT"] * np.array([0, 0, 1]) * left_ankle_rot + skeleton[3, :]

    # HIP_RIGHT
    skeleton[5, :] = bone_lengths["HIP_RIGHT-PELVIS"] * np.array([-1, 0, 0]) * hip_rot + skeleton[0, :]

    # KNEE_RIGHT
    right_leg_rot = create_euler_rotation_matrix(angles[9], angles[10], angles[11]) * hip_rot
    skeleton[6, :] = bone_lengths["KNEE_RIGHT-HIP_RIGHT"] * np.array([0, -1, 0]) * right_leg_rot + skeleton[5, :]

    # ANKLE_RIGHT
    right_knee_rot = create_euler_rotation_matrix(angles[12], 0, 0) * right_leg_rot
    skeleton[7, :] = bone_lengths["ANKLE_RIGHT-KNEE_RIGHT"] * np.array([0, -1, 0]) * right_knee_rot + skeleton[6, :]

    # FOOT_RIGHT
    right_ankle_rot = create_euler_rotation_matrix(angles[13], angles[14], 0) * right_knee_rot
    skeleton[8, :] = bone_lengths["FOOT_RIGHT-ANKLE_RIGHT"] * np.array([0, 0, 1]) * right_ankle_rot + skeleton[7, :]

    # PELVIS to SPINE_NAVEL
    spine_rot = create_euler_rotation_matrix(angles[15], angles[16], angles[17]) * hip_rot
    skeleton[9, :] = bone_lengths["PELVIS-SPINE_NAVEL"] * np.array([0, 1, 0]) * spine_rot + skeleton[0, :]

    # SPINE_NAVEL to SPINE_CEST
    spine_chest_rot = create_euler_rotation_matrix(angles[18], angles[19], angles[20]) * spine_rot
    skeleton[10, :] = bone_lengths["SPINE_NAVEL-SPINE_CHEST"] * np.array([0, 1, 0]) * spine_chest_rot + skeleton[9, :]

    # TODO: Finish skeleton definition
    return skeleton


def complete_angle_vector_with_zeros(angles: np.ndarray) -> np.ndarray:
    angles = list(angles)
    # Fill the restricted angles
    angles.insert(7, 0)
    angles.insert(7, 0)
    angles.insert(11, 0)
    angles.insert(16, 0)
    angles.insert(17, 0)
    angles.insert(20, 0)

    # Fill left foot
    angles.insert(12, 0)
    angles.insert(13, 0)
    angles.insert(14, 0)

    # Fill right foot
    angles.insert(24, 0)
    angles.insert(25, 0)
    angles.insert(26, 0)
    return np.array(angles)


def create_euler_matrices(angles: np.ndarray) -> np.ndarray:
    rotations = []
    for i in range(angles.shape[0] // 3):
        rotation = R.from_euler("xyz", angles[i * 3: (i + 1) * 3]).as_matrix()
        rotations.append(rotation)

    # rotations.insert(4, np.eye(3))
    # rotations.insert(8, np.eye(3))
    return np.array(rotations).reshape((len(rotations), 3, 3))


def create_euler_kinematic_chain(angles: np.ndarray):
    rotations = []
    rotation = R.from_euler("xyz", angles[0:3]).as_matrix()
    rotations.append(rotation)

    # Left side
    for i, j in zip([0, 1, 2], [1, 2, 3]):
        rotation = R.from_euler("xyz", angles[j * 3: (j + 1) * 3]).as_matrix()
        last_rotation = rotations[i]
        rotation = np.matmul(last_rotation, rotation)
        rotations.append(rotation)

    # Right side
    for i, j in zip([0, 4, 5], [4, 5, 6]):
        ang = angles[j * 3: (j + 1) * 3]
        rotation = R.from_euler("xyz", ang).as_matrix()
        last_rotation = rotations[i]
        rotation = np.matmul(last_rotation, rotation)
        rotations.append(rotation)

    # Spine
    rotation = R.from_euler("xyz", angles[21:24]).as_matrix()
    rotations.append(np.matmul(rotations[0],  rotation))

    # Spine chest
    rotation = R.from_euler("xyz", angles[24:]).as_matrix()
    rotations.append(np.matmul(rotations[-1],  rotation))

    # Add missing joints
    rotations.insert(4, np.eye(3))
    rotations.insert(8, np.eye(3))
    return np.array(rotations).reshape((len(rotations), 3, 3))


if __name__ == '__main__':
    gl_dict = {
        "FOOT_LEFT-ANKLE_LEFT": 184,
        "ANKLE_LEFT-KNEE_LEFT": 381,
        "KNEE_LEFT-HIP_LEFT": 400,
        "HIP_LEFT-PELVIS": 100,
        "FOOT_RIGHT-ANKLE_RIGHT": 184,
        "ANKLE_RIGHT-KNEE_RIGHT": 381,
        "KNEE_RIGHT-HIP_RIGHT": 400,
        "HIP_RIGHT-PELVIS": 100,
        "PELVIS-SPINE_NAVEL": 178,
        "SPINE_NAVEL-SPINE_CHEST": 145,
    }
    zeros = np.zeros(23)
    # zeros = np.random.random(23)
    normal_pose = build_skeleton(zeros, gl_dict)

    viewer = MoCapViewer(sphere_radius=0.03, sampling_frequency=30, grid_axis=None)
    viewer.add_skeleton(normal_pose.reshape(1, -1), skeleton_connection=skeleton_connection)  # skeleton_orientations=normal_ori)
    viewer.show_window()
