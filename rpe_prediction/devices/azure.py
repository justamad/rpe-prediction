from rpe_prediction.processing import normalize_signal, find_closest_timestamp, fill_missing_data, filter_dataframe
from .sensor_base import SensorBase
from os.path import join
from enum import Enum

import numpy as np
import pandas as pd
import json
import os


excluded_joints = ["EYE", "EAR", "NOSE", "HANDTIP", "THUMB", "CLAVICLE"]


class JointConfidenceLevel(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2  # Current SDK only goes up to here
    HIGH = 3  # Placeholder for future SDK


class AzureKinect(SensorBase):

    conf_values = {}

    def __init__(self, data_path):
        """
        Constructor for Azure Kinect camera
        @param data_path: path where the csv file resides in
        """
        position_file = join(data_path, "positions_3d.csv")
        if not os.path.exists(position_file):
            raise FileNotFoundError(f"File {position_file} does not exist.")

        df = pd.read_csv(position_file, delimiter=';')
        df.set_index('timestamp', drop=True, inplace=True)

        body_idx_c = df['body_idx'].value_counts()
        df = df[df['body_idx'] == body_idx_c.index[body_idx_c.argmax()]]
        df = df.drop('body_idx', axis=1)

        conf = df.filter(like='(c)')
        conf_c = conf.apply(conf.value_counts).fillna(0)
        AzureKinect.conf_values[data_path] = conf_c

        mask = conf >= JointConfidenceLevel.MEDIUM.value
        df = df[[c for c in df.columns if "(c)" not in c]]
        l_mask = pd.DataFrame(np.repeat(mask.to_numpy(), 3, axis=1), columns=df.columns, index=df.index)
        df = df.where(l_mask, np.NAN)
        df = df.interpolate(method='quadratic', order=4).bfill()

        df = filter_dataframe(df, excluded_joints)

        df.index *= 1e-6  # Convert microseconds to seconds
        df = fill_missing_data(df, 30, method='linear', log=True)
        super().__init__(df, 30)

    def multiply_matrix(self, matrix, translation=np.array([0, 0, 0])):
        """
        Multiply all data joint positions with a matrix and add a translation vector
        @param matrix: the rotation matrix
        @param translation: a translation vector
        """
        data = self._data.to_numpy()
        samples, features = data.shape
        result = matrix * data.reshape(-1, 3).T + translation.reshape(3, 1)
        final_result = result.T.reshape(samples, features)
        data = pd.DataFrame(data=final_result, columns=self._data.columns, index=self._data.index)
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

    @staticmethod
    def get_skeleton_joints():
        skeleton_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "joints.json")
        with open(skeleton_file) as f:
            joints = json.load(f)

        return list(map(lambda x: x.lower(), joints))

    def get_synchronization_signal(self) -> np.ndarray:
        return self._data['pos_spine_navel (y)'].to_numpy()

    def get_synchronization_data(self):
        """
        Get the synchronization data
        @return: tuple with (timestamps, raw_data, acc_data, peaks)
        """
        raw_data = normalize_signal(self.get_synchronization_signal())
        acc_data = normalize_signal(np.gradient(np.gradient(raw_data)))  # Calculate 2nd derivative
        return self._data.index, raw_data, acc_data

    def cut_data_based_on_time(self, start_time, end_time):
        """
        Cut the data based on given start and end time
        @param start_time: start time in seconds
        @param end_time: end time in seconds
        """
        start_idx = find_closest_timestamp(self._data.index, start_time)
        end_idx = find_closest_timestamp(self._data.index, end_time)
        self._data = self._data.iloc[start_idx:end_idx]

    def remove_unnecessary_joints(self):
        """
        Remove unnecessary joints from data frame using the excluded joints
        @return: None
        """
        self._data = filter_dataframe(self._data, excluded_joints)

    def set_timestamps(self, timestamps):
        """
        Set the current timestamps to the given timestamps
        @param timestamps: the new timestamps, has to be of same length
        @return: None
        """
        self._data.index = timestamps

    def __repr__(self):
        """
        String representation of Azure Kinect camera class
        @return: camera name
        """
        return "Azure Kinect"

    @property
    def data(self):
        return self._data

    @property
    def timestamps(self):
        return self._data.index
