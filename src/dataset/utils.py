from typing import Tuple
import pandas as pd


def extract_dataset_input_output(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lst = ["subject", "nr_set", "rpe"]
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
