from typing import Tuple, List
from itertools import compress
from scipy import stats

import numpy as np
import pandas as pd
import math


def split_data_based_on_pseudonyms(
        X: pd.DataFrame,
        y: pd.DataFrame,
        train_p: float = 0.8,
        random_seed: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_mask = get_subject_names_random_split(
        y=y,
        train_p=train_p,
        random_seed=random_seed,
    )

    return (
        X.loc[train_mask].copy(),
        y.loc[train_mask].copy(),
        X.loc[~train_mask].copy(),
        y.loc[~train_mask].copy(),
    )


def split_data_based_on_pseudonyms_multiple_inputs(
        X_data: list,
        y_data: pd.DataFrame,
        train_p: float = 0.8,
        random_seed: int = None,
) -> Tuple[List, pd.DataFrame, List, pd.DataFrame]:
    train_mask = get_subject_names_random_split(
        y=y_data,
        train_p=train_p,
        random_seed=random_seed,
    )

    X_train = list(compress(X_data, train_mask))
    X_test = list(compress(X_data, ~train_mask))
    return (
        X_train,
        y_data.loc[train_mask].copy().reset_index(drop=True),
        X_test,
        y_data.loc[~train_mask].copy().reset_index(drop=True),
    )


def get_subject_names_random_split(
        y: pd.DataFrame,
        train_p: float = 0.7,
        random_seed: int = None,
):
    subject_names = sorted(y["name"].unique())
    nr_subjects = math.ceil(len(subject_names) * train_p)

    if random_seed is not None:
        np.random.seed(random_seed)

    train_subjects = np.random.choice(subject_names, nr_subjects, replace=False)
    train_idx = y["name"].isin(train_subjects)
    return train_idx


def normalize_rpe_values_min_max(
        df: pd.DataFrame,
        digitize: bool = False,
        bins: int = 10,
):
    subjects = df['name'].unique()
    for subject_name in subjects:
        mask = df['name'] == subject_name
        rpe_values = df.loc[mask, 'rpe'].to_numpy()
        rpe_norm = (rpe_values - rpe_values.min()) / (rpe_values.max() - rpe_values.min())

        if digitize:
            rpe_norm = np.digitize(rpe_norm, bins=np.arange(bins) / bins)

        df.loc[mask, 'rpe'] = rpe_norm

    return df


def filter_outliers_z_scores(df: pd.DataFrame):
    z_scores = stats.zscore(df)
    z_scores = z_scores.dropna(axis=1, how='all')
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    return filtered_entries
    # new_df = df[filtered_entries]
    # return new_df
