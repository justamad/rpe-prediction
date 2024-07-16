import pandas as pd
import numpy as np


def calculate_linear_joint_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the linear joint positions of the skeleton according to the paper:
    'Action recognition using kinematics posture feature on 3D skeleton joint locations'
    presented in the journal
    'Pattern Recognition Letters'
    """
    ref_joint = df.loc[:, [c for c in df.columns if "PELVIS" in c]]
    neck_df = df.loc[:, [c for c in df.columns if "NECK" in c]]
    spine_chest_df = df.loc[:, [c for c in df.columns if "SPINE_CHEST" in c]]

    df = df.loc[:, [c for c in df.columns if "PELVIS" not in c]]
    neck_length = np.linalg.norm(neck_df.values - spine_chest_df.values, axis=1).mean()

    n_joints = df.shape[1] // 3
    for j in range(n_joints):
        df.iloc[:, j * 3:(j + 1) * 3] = (df.iloc[:, j * 3:(j + 1) * 3].values - ref_joint.values) / neck_length

    return df


def calculate_skeleton_images(pos_df: pd.DataFrame, ori_df) -> np.ndarray:
    unused_joints = ["WRIST", "FOOT"]
    pos_df.drop("Repetition", axis=1, inplace=True)
    ori_df.drop("Repetition", axis=1, inplace=True)
    # pos_df = pos_df.loc[:, [c for c in pos_df.columns if not any([j in c for j in unused_joints])]]
    # ori_df = ori_df.loc[:, [c for c in ori_df.columns if not any([j in c for j in unused_joints])]]
    # pos_df = calculate_relative_joint_positions(pos_df)
    # disp_df = calculate_displacement(pos_df)
    pos_df = calculate_linear_joint_positions(pos_df)
    # ori_df = ori_df.loc[:, [c for c in ori_df.columns if "PELVIS" not in c]]
    # image = np.stack([pos_df.values, disp_df.values, ori_df.values], axis=2)
    return pos_df.values


def calculate_relative_joint_positions(df: pd.DataFrame, joint: str = "PELVIS") -> pd.DataFrame:
    pelvis = df.loc[:, [c for c in df.columns if joint in c]]
    other = df.loc[:, [c for c in df.columns if joint not in c]]
    n_joints = other.shape[1] // 3
    for j in range(n_joints):
        other.iloc[:, j * 3:(j + 1) * 3] = other.iloc[:, j * 3:(j + 1) * 3].values - pelvis.values

    return other


def calculate_displacement(df: pd.DataFrame) -> pd.DataFrame:
    df = df.diff().fillna(0)
    return df
