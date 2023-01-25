from typing import Tuple
from scipy import stats

import numpy as np
import pandas as pd
import math


def extract_dataset_input_output(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lst = ["subject", "rpe", "nr_rep", "nr_set"]
    inputs = df.drop(lst, axis=1, inplace=False, errors="ignore")
    outputs = df[df.columns.intersection(lst)]
    return inputs, outputs


def normalize_subject_rpe(df: pd.DataFrame) -> pd.DataFrame:
    for subject in df["subject"].unique():
        mask = df["subject"] == subject
        rpe_values = df.loc[mask, "rpe"].to_numpy()
        rpe_norm = (rpe_values - rpe_values.min()) / (rpe_values.max() - rpe_values.min())
        df.loc[mask, "rpe"] = rpe_norm

    return df


def discretize_subject_rpe(df: pd.DataFrame) -> pd.DataFrame:
    labels = df["rpe"]
    labels[labels <= 15] = 0
    labels[(labels > 15) & (labels <= 18)] = 1
    labels[labels > 18] = 2
    df["rpe"] = labels
    return df


def split_data_based_on_pseudonyms(
        X: pd.DataFrame,
        y: pd.DataFrame,
        train_p: float = 0.8,
        random_seed: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_mask = get_subject_names_random_split(
        df=y,
        train_p=train_p,
        random_seed=random_seed,
    )

    return (
        X.loc[train_mask].copy(),
        y.loc[train_mask].copy(),
        X.loc[~train_mask].copy(),
        y.loc[~train_mask].copy(),
    )


def get_subject_names_random_split(
        df: pd.DataFrame,
        train_p: float = 0.7,
        random_seed: int = None,
) -> pd.Series:
    subject_names = sorted(df["subject"].unique())
    nr_subjects = math.ceil(len(subject_names) * train_p)

    if random_seed is not None:
        np.random.seed(random_seed)

    train_subjects = np.random.choice(subject_names, nr_subjects, replace=False)
    train_idx = df["subject"].isin(train_subjects)
    return train_idx


def normalize_rpe_values_min_max(
        df: pd.DataFrame,
        digitize: bool = False,
        bins: int = 10,
):
    subjects = df["name"].unique()
    for subject_name in subjects:
        mask = df["name"] == subject_name
        rpe_values = df.loc[mask, "rpe"].to_numpy()
        rpe_norm = (rpe_values - rpe_values.min()) / (rpe_values.max() - rpe_values.min())

        if digitize:
            rpe_norm = np.digitize(rpe_norm, bins=np.arange(bins) / bins)

        df.loc[mask, "rpe"] = rpe_norm

    return df


def normalize_data_by_subject(df: pd.DataFrame, label_cols: int = 3) -> pd.DataFrame:
    total_df = pd.DataFrame()
    for name, group in df.groupby("subject"):
        sub_df = group.iloc[:, :-label_cols].values
        group.iloc[:, :-label_cols] = (sub_df - sub_df.mean()) / (sub_df.std())
        total_df = pd.concat([total_df, group], ignore_index=True)

    return total_df


def filter_outliers_z_scores(df: pd.DataFrame):
    z_scores = stats.zscore(df.iloc[:, :-4])
    z_scores = z_scores.dropna(axis=1, how="all")
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    return filtered_entries
    # new_df = df[filtered_entries]
    # return new_df
