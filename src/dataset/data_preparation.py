from typing import Tuple, Union, List
from scipy import stats

import numpy as np
import pandas as pd
import logging

from sklearn.feature_selection import VarianceThreshold

META_DATA = ["subject", "set_id", "rpe"]


def extract_dataset_input_output(
        df: pd.DataFrame,
        labels: Union[List[str], str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(labels, str):
        columns = [labels] + META_DATA
    elif isinstance(labels, list):
        columns = labels + META_DATA
    else:
        raise ValueError(f"Unknown ground truth column type: {type(labels)}.")

    for col in columns:
        if col not in df.columns:
            logging.warning(f"Column {col} not in dataframe. Proceeding anyways...")

    inputs = df.drop(columns, axis=1, inplace=False, errors="ignore")
    outputs = df.loc[:, df.columns.intersection(columns)]
    return inputs, outputs


def discretize_subject_rpe(df: pd.DataFrame) -> pd.DataFrame:
    labels = df["rpe"]
    labels[labels <= 15] = 0
    labels[(labels > 15) & (labels <= 18)] = 1
    labels[labels > 18] = 2
    df["rpe"] = labels
    return df


def normalize_labels_min_max(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if label_col not in df.columns:
        raise ValueError(f"Label column {label_col} not in dataframe.")

    if "subject" not in df.columns:
        raise ValueError("Subject column not in dataframe.")

    subjects = df["subject"].unique()
    for subject_name in subjects:
        mask = df["subject"] == subject_name
        values = df.loc[mask, label_col].values
        rpe_norm = (values - values.min()) / (values.max() - values.min())
        df.loc[mask, label_col] = rpe_norm

    return df


def calculate_trend_labels(y: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if label_col not in y.columns:
        raise ValueError(f"Label column {label_col} not in dataframe.")

    if "subject" not in y.columns:
        raise ValueError("Subject column not in dataframe.")

    subjects = y["subject"].unique()
    for subject in subjects:
        mask = y["subject"] == subject
        values = y.loc[mask, label_col].values
        y.loc[mask, label_col] = np.diff(values, prepend=values[0])

    return y


def normalize_data_by_subject(X: pd.DataFrame, y: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
    if method not in ["standard", "min_max"]:
        raise ValueError(f"Unknown normalization method: {method}.")

    for subject in y["subject"].unique():
        mask = (y["subject"] == subject) & (~X.eq(0).all(axis=1))
        if method == "standard":
            X.loc[mask] = (X.loc[mask] - X.loc[mask].mean()) / X.loc[mask].std()
        else:
            X.loc[mask] = (X.loc[mask] - X.loc[mask].min()) / (X.loc[mask].max() - X.loc[mask].min())

    return X


def normalize_data_global(X: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
    mask = ~X.eq(0).all(axis=1)
    if method == "standard":
        X.loc[mask] = (X.loc[mask] - X.loc[mask].mean()) / X.loc[mask].std()
    else:
        X.loc[mask] = (X.loc[mask] - X.loc[mask].min()) / (X.loc[mask].max() - X.loc[mask].min())

    return X


def clip_outliers_z_scores(df: pd.DataFrame, sigma: float = 3.0):
    df[df > sigma] = sigma
    df[df < -sigma] = -sigma
    return df


def filter_labels_outliers_per_subject(
        X: Union[np.ndarray, pd.DataFrame],
        y: pd.DataFrame,
        label_col: str,
        sigma: float = 3.1,
) -> Tuple[Union[np.ndarray, pd.DataFrame], pd.DataFrame]:
    if "subject" not in y.columns:
        raise ValueError("Subject column not in dataframe.")

    final_mask = np.ones(len(y), dtype=bool)
    for subject in y["subject"].unique():
        mask = np.array(y["subject"] == subject)
        new_mask = np.abs(stats.zscore(y.loc[mask, label_col].values)) < sigma
        final_mask[mask] = final_mask[mask] & new_mask

    X = X[final_mask]
    y = y[final_mask]
    return X, y


def drop_correlated_features(X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    X.drop(to_drop, axis=1, inplace=True)
    return X


def add_rolling_statistics(X: pd.DataFrame, y: pd.DataFrame, win: List[int]) -> pd.DataFrame:
    if "subject" not in y.columns:
        raise ValueError("Subject column not in dataframe.")

    subjects = []
    for subject in y["subject"].unique():
        sub_df = X.loc[y["subject"] == subject, :]
        data_frames = []
        for window_size in win:
            mean_df = sub_df.rolling(window=window_size).mean()
            mean_df.rename(columns={c: f"{c}_TM({window_size})" for c in mean_df.columns}, inplace=True)
            std_df = sub_df.rolling(window=window_size).std()
            std_df.rename(columns={c: f"{c}_TS({window_size})" for c in std_df.columns}, inplace=True)
            data_frames.append(mean_df)
            data_frames.append(std_df)

        subject_df = pd.concat(data_frames, axis=1)
        subjects.append(subject_df)

    temp_context_df = pd.concat(subjects, axis=0)
    temp_context_df.fillna(0, inplace=True)
    temp_context_df = normalize_data_by_subject(temp_context_df, y)
    return temp_context_df


def dl_split_data(
        X: np.ndarray,
        y: pd.DataFrame,
        label_col: Union[str, List[str]],
        p_train: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # if isinstance(label_col, str):
        # label_col = [label_col]

    subjects = y["subject"].unique()
    train_subjects = subjects[:int(len(subjects) * p_train)]
    train_mask = y["subject"].isin(train_subjects)

    X_test, y_test = X[~train_mask], y.loc[~train_mask, :]
    X_train, y_train = X[train_mask], y.loc[train_mask, :]

    y_train = y_train[label_col].values
    y_test = y_test[label_col].values
    return X_train, y_train, X_test, y_test


def dl_normalize_data_3d_subject(X: np.ndarray, y: pd.DataFrame, method="min_max"):
    if "subject" not in y.columns:
        raise ValueError("Subject column not in dataframe.")

    if method not in ["min_max", "std"]:
        raise ValueError(f"Unknown normalization method: {method}")

    for subject in y["subject"].unique():
        mask = y["subject"] == subject
        data = X[mask]
        arr = np.vstack(data)

        if method == "min_max":
            minimum = np.min(arr, axis=0)
            maximum = np.max(arr, axis=0)
            for trial in range(len(data)):
                cur_data = (data[trial] - minimum) / (maximum - minimum)
                cur_data = np.clip(cur_data, 0, 1)
                cur_data = np.nan_to_num(cur_data)
                data[trial] = cur_data

        else:
            mean = np.mean(arr, axis=0)
            std = np.std(arr, axis=0)

            for trial in range(len(data)):
                cur_data = (data[trial] - mean) / std
                cur_data = np.clip(cur_data, -3, 3)
                cur_data = np.nan_to_num(cur_data)
                data[trial] = cur_data

        X[mask] = data

    return X


def dl_normalize_data_3d_global(X: np.ndarray, method="min_max"):
    if method not in ["min_max", "std"]:
        raise ValueError(f"Unknown normalization method: {method}")

    arr = np.vstack(X)
    if method == "min_max":
        minimum = np.min(arr, axis=0)
        maximum = np.max(arr, axis=0)
        for skeleton in range(len(X)):
            X[skeleton] = (X[skeleton] - minimum) / (maximum - minimum)

    if method == "std":
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        for skeleton in range(len(X)):
            X[skeleton] = (X[skeleton] - mean) / std

    return X


def remove_low_variance_features(X: pd.DataFrame, threshold: float = 0.01):
    variance_selector = VarianceThreshold(threshold=threshold)
    variance_selector.fit_transform(X)
    return X.loc[:, variance_selector.get_support()]


def get_highest_correlation_features(X: pd.DataFrame, y: np.ndarray, k=300):
    correlations = np.abs(X.corrwith(y))
    selected_features_mask = correlations.nlargest(k).index
    X_selected = X[selected_features_mask]
    return X_selected
