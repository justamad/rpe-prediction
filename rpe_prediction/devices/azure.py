from rpe_prediction.processing import normalize_signal, find_closest_timestamp, fill_missing_data, filter_dataframe
from .sensor_base import SensorBase
from os.path import join

import numpy as np
import pandas as pd
import json
import os


excluded_joints = ["eye", "ear", "nose", "handtip", "thumb", "clavicle"]


class AzureKinect(SensorBase):

    def __init__(self, data_path):
        """
        Constructor for Azure Kinect camera
        @param data_path: path where the csv file resides in
        """
        position_file = join(data_path, "positions_3d.csv")
        orientation_file = join(data_path, "orientations_3d.csv")

        if not os.path.exists(position_file) or not os.path.exists(orientation_file):
            raise FileNotFoundError(f"File {position_file} does not exist.")

        # Read in csv files for position and orientation
        pos_data = pd.read_csv(position_file, delimiter=';')
        ori_data = pd.read_csv(orientation_file, delimiter=';')

        # Remove all other bodies found in the images
        body_idx_counts = pos_data['body_idx'].value_counts()
        most_often_body_idx = body_idx_counts.index[body_idx_counts.argmax()]
        pos_data = pos_data[pos_data['body_idx'] == most_often_body_idx]
        ori_data = ori_data[ori_data['body_idx'] == most_often_body_idx]

        # Remove unnecessary data from data frames
        pos_data = pos_data[[c for c in pos_data.columns if "(c)" not in c and "body_idx" not in c]].copy()
        new_names = [(i, i.lower() + " pos") for i in pos_data.iloc[:, 1:].columns.values]
        pos_data.rename(columns=dict(new_names), inplace=True)

        ori_data = ori_data[[c for c in ori_data.columns if "body_idx" not in c and "timestamp" not in c]].copy()
        new_names = [(i, i.lower() + " ori") for i in ori_data.columns.values]
        ori_data.rename(columns=dict(new_names), inplace=True)
        data = pd.concat([pos_data, ori_data], axis=1)

        super().__init__(data, 30)

    def process_raw_data(self, log=False):
        """
        Processing the raw data
        This consists of timestamps and filling missing data
        """
        self._data.loc[:, self._data.columns == 'timestamp'] *= 1e-6
        self._data = fill_missing_data(self._data, self.sampling_frequency, log=log)
        # self._data = self._data.set_index('timestamp')

    def multiply_matrix(self, matrix, translation=np.array([0, 0, 0])):
        """
        Multiply all data joint positions with a matrix and add a translation vector
        @param matrix: the rotation matrix
        @param translation: a translation vector
        """
        df = self._data.filter(regex='pos').copy()
        data = df.to_numpy()
        samples, features = data.shape
        result = matrix * data.reshape(-1, 3).T + translation.reshape(3, 1)
        final_result = result.T.reshape(samples, features)
        data = pd.DataFrame(data=final_result, columns=df.columns, index=df.index)
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

    @property
    def position_data(self):
        data = self._data.filter(regex='pos').copy()
        new_names = [(i, i.replace('pos', '')) for i in data.columns.values]
        data.rename(columns=dict(new_names), inplace=True)
        return data

    def __repr__(self):
        """
        String representation of Azure Kinect camera class
        @return: camera name
        """
        return "Azure Kinect"
