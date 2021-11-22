from .geometry import (
    calculate_angle_in_radians_between_vectors,
    create_rotation_matrix_y_axis,
    apply_affine_transformation,
    create_rotation_matrix_z_axis,
)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors


def get_all_columns_for_joint(df, joint):
    columns = [col for col in df.columns if joint.lower() in col.lower()]
    if not columns:
        raise Exception(f"Cannot find joint: {joint} in {df}")

    return df[columns]


def remove_columns_from_dataframe(df: pd.DataFrame, excluded_matches: list):
    for excluded_part in excluded_matches:
        df = df.loc[:, ~df.columns.str.contains(excluded_part)]
    return df


def get_joint_names_from_columns_as_list(df, joints):
    return list(filter(lambda x: any([c for c in df.columns if x in c]), joints))


def get_hsv_color(cur_value, max_value):
    return matplotlib.colors.hsv_to_rgb([cur_value / max_value * 0.75, 1, 1])


def compute_mean_and_std_of_joint_for_subjects(subject_iterator):
    trials_for_subjects = {}

    for set_data in subject_iterator:
        subject_name = set_data['subject_name']
        df = pd.read_csv(set_data['azure'], sep=';').set_index('timestamp', drop=True)
        if subject_name not in trials_for_subjects:
            trials_for_subjects[subject_name] = [df]
        else:
            trials_for_subjects[subject_name].append(df)

    means = {k: pd.concat(v, ignore_index=True).mean(axis=0) for k, v in trials_for_subjects.items()}
    std_devs = {k: pd.concat(v, ignore_index=True).std(axis=0) for k, v in trials_for_subjects.items()}
    return means, std_devs


def check_angle_between_x_axis(df: pd.DataFrame):
    foot_left = get_all_columns_for_joint(df, "FOOT_LEFT").to_numpy()
    foot_right = get_all_columns_for_joint(df, "FOOT_RIGHT").to_numpy()
    v1 = foot_left - foot_right
    v2 = np.repeat(np.array([1, 0, 0]).reshape(1, 3), len(v1), axis=0)
    return calculate_angle_in_radians_between_vectors(v1, v2)


def align_skeleton_parallel_to_x_axis(
        df: pd.DataFrame,
        show: bool = False
) -> pd.DataFrame:
    angles_y = []
    angles_z = []

    for angle in range(360):
        angle_y = check_angle_between_x_axis(apply_affine_transformation(df, create_rotation_matrix_y_axis(angle)))
        angle_z = check_angle_between_x_axis(apply_affine_transformation(df, create_rotation_matrix_z_axis(angle)))

        angles_y.append(angle_y.mean())
        angles_z.append(angle_z.mean())

    rot_angle_y = np.argmin(angles_y)
    rot_angle_z = np.argmin(angles_z)

    if show:
        plt.plot(angles_y, label="Angles Y")
        plt.plot(angles_z, label="Angles Z")
        plt.xlabel("Degrees")
        plt.ylabel("Error [mm]")
        plt.title(f"Y: {rot_angle_y}, dev: {angles_y[rot_angle_y]}, Z:{rot_angle_z}, dev: {angles_z[rot_angle_z]}")
        plt.show()

    df = apply_affine_transformation(df, create_rotation_matrix_y_axis(float(rot_angle_y)))
    df = apply_affine_transformation(df, create_rotation_matrix_z_axis(float(rot_angle_z)))
    return df
