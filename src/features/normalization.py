import numpy as np
import pandas as pd


def normalize_skeleton_positions(
        df: pd.DataFrame,
        origin_joint: str = "PELVIS"
) -> pd.DataFrame:
    if df.shape[1] % 3 != 0:
        raise AttributeError(f"Columns of dataframe should be a multiple of 3. Got {df.shape[1]}")

    n_joints = df.shape[1] // 3

    origin = df[[c for c in df.columns if origin_joint in c]].to_numpy()
    origin = np.tile(origin, (1, n_joints))
    df.loc[:, :] = df.to_numpy() - origin
    return df
