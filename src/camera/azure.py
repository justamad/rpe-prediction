from src.processing import (
    find_closest_timestamp,
    identify_and_fill_gaps_in_data,
    remove_columns_from_dataframe,
    apply_affine_transformation,
    get_all_columns_for_joint,
)

from collections import OrderedDict
from enum import Enum

import numpy as np
import pandas as pd
import json
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
            raise FileNotFoundError(f"File {csv_file} does not exist.")

        df = pd.read_csv(csv_file, delimiter=';').set_index('timestamp', drop=True)

        body_idx_c = df['body_idx'].value_counts()
        df = df[df['body_idx'] == body_idx_c.index[body_idx_c.argmax()]]
        df = df.drop('body_idx', axis=1)

        conf = df.filter(like='(c)')
        conf_c = conf.apply(conf.value_counts).fillna(0)
        AzureKinect.conf_values[csv_file] = conf_c

        mask = conf >= JointConfidenceLevel.MEDIUM.value
        df = df[[c for c in df.columns if "(c)" not in c]]
        l_mask = pd.DataFrame(np.repeat(mask.to_numpy(), 3, axis=1), columns=df.columns, index=df.index)
        df = df.where(l_mask, np.NAN)
        df = df.interpolate(method='quadratic', order=4).bfill()

        df = remove_columns_from_dataframe(df, excluded_joints)

        df.index *= 1e-6  # Convert microseconds to seconds
        df = identify_and_fill_gaps_in_data(df, 30, method='linear', log=False)
        self._data = df

    def apply_affine_transformation(self, matrix: np.ndarray, translation: np.ndarray = np.array([0, 0, 0])):
        new_df = apply_affine_transformation(self._data, matrix, translation)
        self._data.update(new_df)

    def __getitem__(self, item: str):
        return get_all_columns_for_joint(self._data, item)

    @staticmethod
    def get_skeleton_joints():
        skeleton_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "joints.json")
        with open(skeleton_file) as f:
            joints = json.load(f)

        return list(map(lambda x: x.lower(), joints))

    def cut_data_based_on_time_stamps(self, start_time, end_time):
        start_idx = find_closest_timestamp(self._data.index, start_time)
        end_idx = find_closest_timestamp(self._data.index, end_time)
        self._data = self._data.iloc[start_idx:end_idx]

    def remove_unnecessary_joints(self):
        self._data = remove_columns_from_dataframe(self._data, excluded_joints)

    def set_new_timestamps(self, timestamps):
        self._data.index = timestamps

    def add_delta_to_timestamps(self, delta):
        self._data.index += delta

    def cut_data_by_index(self, start: int = 0, end: int = -1):
        self._data = self._data.iloc[start:end]

    def cut_data_by_label(self, start, end):
        self._data = self._data.loc[start:end]

    def __repr__(self):
        return "Azure Kinect"

    @property
    def data(self):
        return self._data

    @property
    def timestamps(self):
        return self._data.index

    @staticmethod
    def read_skeleton_definition_as_tuple(df: pd.DataFrame):
        json_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "azure.json")
        joints = AzureKinect.get_joints_as_list(df)
        with open(json_file) as f:
            connections = json.load(f)

        skeleton = []
        for j1, j2 in connections:
            if j1 not in joints or j2 not in joints:
                continue
            skeleton.append((joints.index(j1), joints.index(j2)))

        return skeleton

    @staticmethod
    def get_joints_as_list(df: pd.DataFrame):
        columns = [column.replace(column[column.find(" ("):column.find(")") + 1], "") for column in df.columns]
        return list(OrderedDict.fromkeys(columns))