from typing import List

import numpy as np
import pandas as pd


def zero_pad_data_frame(df: pd.DataFrame, max_length: int) -> pd.DataFrame:
    if len(df) > max_length:
        raise AttributeError("Data frame is longer than max length")

    if len(df) == max_length:
        return df

    rows_to_pad = max_length - len(df)
    resample = np.pad(df, pad_width=((0, rows_to_pad), (0, 0)), mode="constant", constant_values=0)
    return pd.DataFrame(resample, columns=df.columns)


def mask_repetitions(df: pd.DataFrame, repetitions: List, col_name: str = "Repetition") -> pd.DataFrame:
    df[col_name] = -1
    for idx, (p1, p2) in enumerate(repetitions):
        df.loc[(df.index >= p1) & (df.index <= p2), col_name] = idx
    return df


def impute_dataframe(df: pd.DataFrame):
    df = df.interpolate(method="linear", limit_direction="forward", axis=0)
    df = df.fillna(0)
    return df


# def truncate_data_frames(*data_frames) -> List[pd.DataFrame]:
#     start_time = max([df.index[0] for df in data_frames])
#     end_time = min([df.index[-1] for df in data_frames])
#     result = [df[(df.index >= start_time) & (df.index < end_time)] for df in data_frames]
#     return result