import pandas as pd
import numpy as np
import matplotlib.colors


def remove_columns_from_dataframe(df: pd.DataFrame, excluded_matches: list):
    for excluded_part in excluded_matches:
        df = df.loc[:, ~df.columns.str.contains(excluded_part)]
    return df


def get_joint_names_from_columns_as_list(df, joints):
    return list(filter(lambda x: any([c for c in df.columns if x in c]), joints))


def get_hsv_color_interpolation(cur_value, max_value):
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
