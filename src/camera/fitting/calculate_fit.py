from .solver import fit_individual_frame
from .kinematic_model import skeleton_connection, complete_angle_vector_with_zeros
from typing import Dict
from tqdm import tqdm
from PyMoCapViewer import MoCapViewer

import pandas as pd
import numpy as np


joint_connections = [
    # Left side
    ("FOOT_LEFT", "ANKLE_LEFT"),
    ("ANKLE_LEFT", "KNEE_LEFT"),
    ("KNEE_LEFT", "HIP_LEFT"),
    ("HIP_LEFT", "PELVIS"),
    # Right side
    ("FOOT_RIGHT", "ANKLE_RIGHT"),
    ("ANKLE_RIGHT", "KNEE_RIGHT"),
    ("KNEE_RIGHT", "HIP_RIGHT"),
    ("HIP_RIGHT", "PELVIS"),
    # Upper Body -> Spine
    ("PELVIS", "SPINE_NAVEL"),
    ("SPINE_NAVEL", "SPINE_CHEST"),
]


def calculate_skeleton_orientations(df: pd.DataFrame):
    bone_lengths = calculate_bone_segments(df)
    df = restructure_data_frame(df)
    # start, end = 0, 200
    # df = df.iloc[start:end, :]

    fitted_skeleton = []
    fitted_angles = []
    opt_angles = np.zeros(21)

    for frame in tqdm(range(df.shape[0])):
        pos, opt_angles = fit_individual_frame(
            frame=df.iloc[frame].values.T.reshape(-1, 3),
            bone_lengths=bone_lengths,
            x0_angles=opt_angles,
        )
        fitted_skeleton.append(pos.reshape(-1))
        fitted_angles.append(complete_angle_vector_with_zeros(opt_angles))

    fitted_skeleton = pd.DataFrame(data=np.array(fitted_skeleton), columns=df.columns)
    fitted_angles = pd.DataFrame(data=np.array(fitted_angles), columns=df.columns)
    return fitted_skeleton, fitted_angles


def calculate_bone_segments(df: pd.DataFrame) -> Dict:
    bone_lengths = {}

    for j1, j2 in joint_connections:
        marker_1 = df[[f"{j1} ({c})" for c in ["x", "y", "z"]]].values
        marker_2 = df[[f"{j2} ({c})" for c in ["x", "y", "z"]]].values

        lengths = np.linalg.norm(marker_1 - marker_2, axis=1)
        mean, std = lengths.mean(), lengths.std()
        bone_lengths[f"{j1}-{j2}"] = mean

    return bone_lengths


def restructure_data_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["PELVIS", "HIP_LEFT", "KNEE_LEFT", "ANKLE_LEFT", "FOOT_LEFT", "HIP_RIGHT", "KNEE_RIGHT", "ANKLE_RIGHT",
               "FOOT_RIGHT", "SPINE_NAVEL", "SPINE_CHEST"]
    columns = [f"{c} ({d})" for c in columns for d in ["x", "y", "z"]]

    df_new = df[columns]
    return df_new


if __name__ == '__main__':
    test_df = pd.read_csv("../../fused.csv", index_col=0)
    new = True

    if new:
        positions, orientations = calculate_skeleton_orientations(test_df)
        positions.to_csv("positions.csv", sep=";")
        orientations.to_csv("orientation.csv", sep=";")
    else:
        positions = pd.read_csv("positions.csv", index_col=0, sep=";")
        orientations = pd.read_csv("orientation.csv", index_col=0, sep=";")
        print(positions.shape)
        print(orientations.shape)

    viewer = MoCapViewer(sphere_radius=0.015, grid_axis=None)
    viewer.add_skeleton(
        positions,
        skeleton_orientations=orientations,
        orientation="euler",
        skeleton_connection=skeleton_connection,
    )
    # viewer.add_skeleton(test_df)
    viewer.show_window()
