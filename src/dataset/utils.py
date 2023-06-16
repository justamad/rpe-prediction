import numpy as np
import pandas as pd

from typing import List


def zero_pad_dataset(X: np.ndarray, max_length) -> np.ndarray:
    sequences = []
    for seq in range(len(X)):
        sequences.append(zero_pad_array(X[seq], max_length))

    X = np.array(sequences)
    X = np.nan_to_num(X)
    return X


def zero_pad_array(array: np.ndarray, max_length: int) -> np.ndarray:
    if len(array) > max_length:
        raise AttributeError("Data frame is longer than max length")

    if len(array) == max_length:
        return array

    rows_to_pad = max_length - len(array)
    resample = np.pad(array, pad_width=((0, rows_to_pad), (0, 0), (0, 0)), mode="constant", constant_values=0)
    return resample


def mask_repetitions(df: pd.DataFrame, repetitions: List, col_name: str = "Repetition") -> pd.DataFrame:
    df[col_name] = -1
    for idx, (p1, p2) in enumerate(repetitions):
        df.loc[(df.index >= p1) & (df.index <= p2), col_name] = idx
    return df


def impute_dataframe(df: pd.DataFrame):
    df = df.interpolate(method="linear", limit_direction="forward", axis=0)
    df = df.fillna(0)
    return df
