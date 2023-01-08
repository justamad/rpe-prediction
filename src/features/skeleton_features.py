from src.processing import calculate_angle_in_radians_between_vectors

import pandas as pd
import numpy as np

joints_with_reference = [("HIP_RIGHT", "KNEE_RIGHT"),
                         ("HIP_LEFT", "KNEE_LEFT"),
                         ("KNEE_RIGHT", "ANKLE_RIGHT"),
                         ("KNEE_LEFT", "ANKLE_LEFT"),
                         ("SHOULDER_RIGHT", "ELBOW_RIGHT"),
                         ("SHOULDER_LEFT", "ELBOW_LEFT"),
                         ("SHOULDER_LEFT", "SHOULDER_RIGHT"),
                         ("HIP_LEFT", "HIP_RIGHT"),
                         ("KNEE_LEFT", "KNEE_RIGHT"),
                         ("ELBOW_LEFT", "ELBOW_RIGHT")]

joints_with_first_joint_as_origin = [("SPINE_CHEST", "NECK", "SPINE_NAVEL"),
                                     ("SPINE_NAVEL", "SPINE_CHEST", "PELVIS"),
                                     ("HIP_LEFT", "PELVIS", "KNEE_LEFT"),
                                     ("HIP_RIGHT", "PELVIS", "KNEE_RIGHT"),
                                     ("KNEE_RIGHT", "HIP_RIGHT", "ANKLE_RIGHT"),
                                     ("KNEE_LEFT", "HIP_LEFT", "ANKLE_LEFT")]


def calculate_1d_joint_velocities(df: pd.DataFrame) -> pd.DataFrame:
    diff = df.diff(axis=0).dropna(axis="index")
    diff = diff.add_suffix('_1D_VELOCITY')
    return diff


def calculate_3d_joint_velocities(df: pd.DataFrame) -> pd.DataFrame:
    diff = df.diff(axis=0).dropna(axis="index")
    euclidean_distances = np.sqrt(np.sum((diff.to_numpy() ** 2).reshape(-1, 3), axis=1))
    gradients = euclidean_distances.reshape(diff.shape[0], diff.shape[1] // 3)
    return pd.DataFrame(
        data=gradients,
        columns=list(map(lambda x: x[:-4] + "_3D_VELOCITY", df.columns[::3])),
        index=diff.index,
    )


def calculate_joint_angles_with_reference_joint(
        df: pd.DataFrame,
        reference: str = "PELVIS",
) -> pd.DataFrame:
    r = df[[c for c in df.columns if reference in c]].to_numpy()
    result = []
    columns = []

    for j1, j2 in joints_with_reference:
        p1 = df[[c for c in df.columns if j1 in c]].to_numpy()
        p2 = df[[c for c in df.columns if j2 in c]].to_numpy()
        angles = calculate_angle_in_radians_between_vectors(r - p1, p2 - r)
        result.append(angles)
        columns.append(f"{j1}={j2}")

    return pd.DataFrame(
        data=np.stack(result, axis=-1),
        columns=columns,
        index=df.index,
    )


def calculate_angles_between_3_joints(df: pd.DataFrame) -> pd.DataFrame:
    data = {}
    for j1, j2, j3 in joints_with_first_joint_as_origin:
        p1 = df[[c for c in df.columns if j1 in c]].to_numpy()
        p2 = df[[c for c in df.columns if j2 in c]].to_numpy()
        p3 = df[[c for c in df.columns if j3 in c]].to_numpy()
        data[f"{j1}->{j2}={j1}->{j3}"] = calculate_angle_in_radians_between_vectors(p2 - p1, p3 - p1)

    return pd.DataFrame(data=data, index=df.index)


def calculate_relative_coordinates_with_reference_joint(
        df: pd.DataFrame,
        reference_joint: str = "PELVIS",
) -> pd.DataFrame:
    n_joints = df.shape[1] // 3
    ref_data = df[[c for c in df.columns if reference_joint in c]].to_numpy()
    ref_data = np.tile(ref_data, n_joints)

    data = df.to_numpy() - ref_data
    new_df = pd.DataFrame(
        data=data,
        columns=[c + "_REL_POSITION" for c in df.columns],
        index=df.index
    )
    new_df.drop(list(new_df.filter(regex=reference_joint)), axis=1, inplace=True)
    return new_df
