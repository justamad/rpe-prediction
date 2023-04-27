import pandas as pd
import numpy as np


def calculate_acceleration_from_position(df: pd.DataFrame) -> pd.DataFrame:
    diff = df.diff(axis=0).diff(axis=0).dropna(axis="index")
    # diff = diff.add_suffix('_1D_VELOCITY')
    return diff


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
