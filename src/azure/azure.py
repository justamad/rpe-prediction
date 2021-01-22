import pandas as pd
import numpy as np
import os
import json


class AzureKinect(object):

    def __init__(self, data_path):
        if isinstance(data_path, pd.DataFrame):
            self.data = data_path
        elif os.path.exists(data_path):
            self.data = pd.read_csv(data_path, delimiter=';')
        else:
            raise Exception(f"Path {data_path} does not exists!")

    def process_raw_data(self, sampling_rate=30):
        # Remove the confidence values and body idx from data frame
        self.data = self.data.loc[:, ~self.data.columns.str.contains('(c)')].copy()
        self.data = self.data.loc[:, ~self.data.columns.str.contains('body_idx')]

        self.data = self.fill_missing_data(self.data)
        # Convert timestamp to seconds and upsample data
        # self.data.loc[:, self.data.columns == 'timestamp'] *= 1e-6

        # Remove timestamp column from data frame
        # self.data = self.data.loc[:, ~self.data.columns.str.contains('timestamp')]

    def fill_missing_data(self, data):
        _, cols = data.shape
        data_body = data.to_numpy()
        epsilon = np.diff(data["timestamp"])
        num_of_missing = (epsilon / 33333 - 1).astype(np.uint8)  # Possible Bug in here!
        print(f'Number of missing data points: {np.sum(num_of_missing)}')

        inc = 0
        for idx, missing_frames in enumerate(num_of_missing):
            if missing_frames <= 0:
                continue

            for j in range(missing_frames):
                data_body = np.insert(data_body, idx + inc + j + 1, np.full(cols, np.nan), axis=0)

            inc += missing_frames

        data = pd.DataFrame(data_body, columns=data.columns).interpolate(method='quadratic')
        return data

    def __getitem__(self, item):
        """
        Get columns that contains the sub-string provided in item
        :param item: given joint name as string
        :return: pandas data frame most likely as nx3 (x,y,z) data frame
        """
        if type(item) is not str:
            raise ValueError(f"Wrong Type for Index. Expected: str, Given: {type(item)}")

        columns = [col for col in self.data.columns if item.lower() in col.lower()]
        if not columns:
            raise Exception(f"Cannot find joint: {item} in {self}")

        return self.data[columns]

    def get_data(self, with_timestamps=False):
        """
        Return the processed skeleton data without timestamps
        :return: Nd-array that contains data with or without timestamps
        """
        if with_timestamps:
            if 'timestamp' not in self.data:
                raise Exception(f"Data for {self} does not contain any timestamps.")
            return self.data.to_numpy()

        if 'timestamp' not in self.data:
            return self.data.to_numpy()
        return self.data.to_numpy()[:, 1:]

    def get_joints_as_list(self):
        """
        Return all joints in a list by removing the duplicate (x,y,z) axes
        :return: list of joint names
        """
        columns = list(self.data.columns)
        if 'timestamp' in self.data:
            columns = columns[1:]

        joints = []
        excluded_chars = ['(x)', '(y)', '(z)', ':x', ':y', ':z']
        for joint in map(lambda x: x.lower(), columns[::3]):
            for ex_char in excluded_chars:
                joint = joint.replace(ex_char, '')
            joints.append(joint.strip().lower())

        return joints

    def get_skeleton_connections(self, json_file):
        joints = self.get_joints_as_list()
        with open(json_file) as f:
            connections = json.load(f)

        return [(joints.index(j1.lower()), joints.index(j2.lower())) for j1, j2 in connections]

    def __repr__(self):
        return "azure"
