from typing import Tuple, List
from itertools import compress

import numpy as np
import pandas as pd
import math


def split_data_based_on_pseudonyms(
        X: pd.DataFrame,
        y: pd.DataFrame,
        train_p: float = 0.8,
        random_seed: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_mask = get_subject_names_random_split(y, train_p, random_seed)
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
    train_mask = get_subject_names_random_split(y_data, train_p, random_seed)
    X_train = list(compress(X_data, train_mask))
    X_test = list(compress(X_data, ~train_mask))
    return (
        X_train,
        y_data.loc[train_mask].copy().reset_index(drop=True),
        X_test,
        y_data.loc[~train_mask].copy().reset_index(drop=True),
    )


def get_subject_names_random_split(y: pd.DataFrame, train_p: float = 0.7, random_seed: int = None):
    subject_names = sorted(y['name'].unique())
    nr_subjects = math.ceil(len(subject_names) * train_p)

    if random_seed is not None:
        np.random.seed(random_seed)

    train_subjects = np.random.choice(subject_names, nr_subjects, replace=False)
    train_idx = y['name'].isin(train_subjects)
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


def normalize_features_z_score(
        df: pd.DataFrame,
        z_score: int = 3,
):
    mean = df.mean()
    std_dev = df.std()
    df = (df - mean) / (z_score * std_dev)
    df = df.clip(-1.0, 1.0)
    return df
