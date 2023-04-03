import logging
from typing import Tuple, Union, List
from scipy import stats

import numpy as np
import pandas as pd
import math

META_DATA = ["subject", "rep_id", "set_id", "rpe"]


def extract_dataset_input_output(
        df: pd.DataFrame,
        labels: Union[List[str], str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(labels, str):
        columns = META_DATA + [labels]
    elif isinstance(labels, list):
        columns = META_DATA + labels
    else:
        raise ValueError(f"Unknown ground truth column type: {type(labels)}.")

    for col in columns:
        if col not in df.columns:
            logging.warning(f"Column {col} not in dataframe. Proceeding anyways...")

    inputs = df.drop(columns, axis=1, inplace=False, errors="ignore")
    outputs = df.loc[:, df.columns.intersection(columns)]
    return inputs, outputs


def normalize_gt_per_subject_mean(df: pd.DataFrame, column: str, normalization: str) -> pd.DataFrame:
    for subject in df["subject"].unique():
        mask = df["subject"] == subject
        gt_values = df.loc[mask, column].to_numpy()
        if normalization == "mean":
            df.loc[mask, column] = (gt_values - gt_values.mean()) / gt_values.std()
        elif normalization == "min_max":
            df.loc[mask, column] = (gt_values - gt_values.min()) / (gt_values.max() - gt_values.min())
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

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
        random_seed: int = 17,
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


def normalize_data_by_subject(X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    for subject in y["subject"].unique():
        mask = (y["subject"] == subject) & (~X.eq(0).all(axis=1))
        X.loc[mask] = (X.loc[mask] - X.loc[mask].mean()) / X.loc[mask].std()

    return X


def normalize_data_global(X: pd.DataFrame) -> pd.DataFrame:
    mask = ~X.eq(0).all(axis=1)
    X.loc[mask] = (X.loc[mask] - X.loc[mask].mean()) / X.loc[mask].std()
    return X


def filter_outliers_z_scores(df: pd.DataFrame):
    z_scores = stats.zscore(df.iloc[:, :-4])
    z_scores = z_scores.dropna(axis=1, how="all")
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    return filtered_entries
