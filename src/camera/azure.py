from enum import Enum

from src.processing import (
    find_closest_timestamp,
    identify_and_fill_gaps_in_data,
    remove_columns_from_dataframe,
    apply_affine_transformation,
    get_all_columns_for_joint,
)

import numpy as np
import pandas as pd
import os

excluded_joints = ["EYE", "EAR", "NOSE", "HANDTIP", "THUMB", "CLAVICLE", "HAND"]


class JointConfidenceLevel(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2  # Current SDK only goes up to here
    HIGH = 3  # Placeholder for future SDK


class AzureKinect(object):
    conf_values = {}

    def __init__(self, csv_file: str):
        if not os.path.exists(csv_file):
            print(os.getcwd())
            raise FileNotFoundError(f"File {csv_file} does not exist.")

        df = pd.read_csv(csv_file, delimiter=";", index_col="timestamp")
        df.index *= 1e-6  # Convert microseconds to seconds

        # Filter other persons in video
        body_idx_c = df["body_idx"].value_counts()
        df = df[df["body_idx"] == body_idx_c.index[body_idx_c.argmax()]]
        df = df.drop("body_idx", axis=1)

        # Check confidence values
        conf = df.filter(like="(c)")
        conf_c = conf.apply(conf.value_counts).fillna(0)
        AzureKinect.conf_values[csv_file] = conf_c

        # mask = conf >= JointConfidenceLevel.MEDIUM.value
        df = df[[c for c in df.columns if "(c)" not in c]]
        # l_mask = pd.DataFrame(np.repeat(mask.to_numpy(), 3, axis=1), columns=df.columns, index=df.index)
        # df = df.where(l_mask, np.NAN)
        # df = df.interpolate(method="quadratic", order=4).bfill()

        df = identify_and_fill_gaps_in_data(df, sampling_rate=30, log=True)
        df = df.fillna(0)
        self._df = df

    def apply_affine_transformation(
            self,
            matrix: np.ndarray,
            translation: np.ndarray = np.array([0, 0, 0]),
    ):
        new_df = apply_affine_transformation(self._df, matrix, translation)
        self._df.update(new_df)

    def __getitem__(self, item: str):
        return get_all_columns_for_joint(self._df, item)

    def cut_data_based_on_time_stamps(self, start_time, end_time):
        start_idx = find_closest_timestamp(self._df.index, start_time)
        end_idx = find_closest_timestamp(self._df.index, end_time)
        self._df = self._df.iloc[start_idx:end_idx]

    def remove_unnecessary_joints(self):
        self._df = remove_columns_from_dataframe(self._df, excluded_joints)

    def set_new_timestamps(self, timestamps):
        self._df.index = timestamps

    def add_delta_to_timestamps(self, delta):
        self._df.index += delta

    def cut_data_by_index(self, start: int = 0, end: int = -1):
        self._df = self._df.iloc[start:end]

    def cut_data_by_label(self, start, end):
        self._df = self._df.loc[start:end]

    def __repr__(self):
        return "Azure Kinect"

    @property
    def data(self):
        return self._df

    @property
    def timestamps(self):
        return self._df.index
