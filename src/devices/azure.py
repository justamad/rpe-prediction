from .processing import normalize_signal, find_closest_timestamp, fill_missing_data
from .sensor_base import SensorBase
from os.path import join

import pandas as pd
import numpy as np
import os
import json


class AzureKinect(SensorBase):

    def __init__(self, data_path, sampling_frequency=30):
        if isinstance(data_path, pd.DataFrame):
            data = data_path
        elif isinstance(data_path, str):
            position_file = join(data_path, "positions_3d.csv")
            orientation_file = join(data_path, "orientations_3d.csv")

            if not os.path.exists(position_file) or not os.path.exists(orientation_file):
                raise FileNotFoundError(f"Given files in {data_path} do not exist.")

            pos_data = pd.read_csv(position_file, delimiter=';')
            pos_data = pos_data[[c for c in pos_data.columns if "(c)" not in c and "body_idx" not in c]].copy()
            new_names = [(i, 'pos_' + i.lower()) for i in pos_data.iloc[:, 1:].columns.values]
            pos_data.rename(columns=dict(new_names), inplace=True)

            ori_data = pd.read_csv(orientation_file, delimiter=';')
            ori_data = ori_data[[c for c in ori_data.columns if "body_idx" not in c and "timestamp" not in c]].copy()
            new_names = [(i, 'ori_' + i.lower()) for i in ori_data.columns.values]
            ori_data.rename(columns=dict(new_names), inplace=True)
            data = pd.concat([pos_data, ori_data], axis=1)
        else:
            raise Exception(f"Unknown argument {data_path} for Azure Kinect class.")

        super().__init__(data, sampling_frequency)

    def process_raw_data(self):
        """
        Processing the raw data
        """
        self._data.loc[:, self._data.columns == 'timestamp'] *= 1e-6
        self._data = fill_missing_data(self._data, self.sampling_frequency, log=True)

    def multiply_matrix(self, matrix, translation=np.array([0, 0, 0])):
        """
        Multiply all data joint positions with a matrix and add a translation vector
        @param matrix: the rotation matrix
        @param translation: a translation vector
        """
        df = self._data.filter(regex='pos_').copy()
        data = df.to_numpy()
        samples, features = data.shape
        result = matrix * data.reshape(-1, 3).T + translation.reshape(3, 1)
        final_result = result.T.reshape(samples, features)
        data = pd.DataFrame(data=final_result, columns=df.columns)
        self._data.update(data)

    def __getitem__(self, item: str):
        """
        Get columns that contains the sub-string provided in item
        @param item: given joint name as string
        @return: pandas data frame most likely as nx3 (x,y,z) data frame
        """
        columns = [col for col in self._data.columns if item.lower() in col.lower()]
        if not columns:
            raise Exception(f"Cannot find joint: {item} in {self}")

        return self._data[columns]

    def get_skeleton_connections(self, json_file: str):
        """
        Returns the joint connections from given json file accordingly to the current joints
        @param json_file: file that contains all skeleton connections
        @return: list that holds tuples (j1, j2) for joint connections (bones)
        """
        joints = self.get_joints_as_list(self.position_data)
        with open(json_file) as f:
            connections = json.load(f)

        return [(joints.index(j1.lower()), joints.index(j2.lower())) for j1, j2 in connections]

    def get_synchronization_signal(self) -> np.ndarray:
        return self._data['pos_spine_navel (y)'].to_numpy()

    def get_synchronization_data(self):
        """
        Get the synchronization data
        @return: tuple with (timestamps, raw_data, acc_data, peaks)
        """
        raw_data = normalize_signal(self.get_synchronization_signal())
        acc_data = normalize_signal(np.gradient(np.gradient(raw_data)))  # Calculate 2nd derivative
        return self.timestamps, raw_data, acc_data

    def cut_data_based_on_time(self, start_time, end_time):
        """
        Cut the data based on given start and end time
        @param start_time: start time in seconds
        @param end_time: end time in seconds
        """
        start_idx = find_closest_timestamp(self.timestamps, start_time)
        end_idx = find_closest_timestamp(self.timestamps, end_time)
        self._data = self._data.iloc[start_idx:end_idx]

    @staticmethod
    def get_joints_as_list(df):
        return list(set([c[:-4] for c in df.columns]))  # Remove axis (ori_test (x))

    @property
    def position_data(self):
        data = self._data.filter(regex='pos_').copy()
        new_names = [(i, i.replace('pos_', '')) for i in data.columns.values]
        data.rename(columns=dict(new_names), inplace=True)
        return data

    @property
    def orientation_data(self):
        data = self._data.filter(regex='ori_').copy()
        new_names = [(i, i.replace('ori_', '')) for i in data.columns.values]
        data.rename(columns=dict(new_names), inplace=True)
        return data

    def __repr__(self):
        """
        String representation of Azure Kinect camera class
        @return: camera name
        """
        return "Azure Kinect"
