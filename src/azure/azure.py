from scipy import interpolate
from src.processing import apply_butterworth_filter, normalize_signal, find_peaks


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

        self.height = 0.5
        self.prominence = 1.5
        self.distance = 40
        self._sampling_frequency = 30

    def process_raw_data(self, sampling_rate=30):
        # Remove the confidence values and body idx from data frame
        self.data = self.data.loc[:, ~self.data.columns.str.contains('(c)')].copy()
        self.data = self.data.loc[:, ~self.data.columns.str.contains('body_idx')]

        self.data = self.fill_missing_data(self.data)
        # Convert timestamp to seconds and upsample data
        # self.data.loc[:, self.data.columns == 'timestamp'] *= 1e-6

        # if sampling_rate != 30:
            # self.data = self.sample_data_uniformly(self.data, sampling_rate)

    def sample_data_uniformly(self, data_frame, sampling_rate):
        """
        Applies a uniform sampling to given data frame
        :param data_frame: data frame consisting the data
        :param sampling_rate: desired sampling frequency
        :return: data frame with filtered data
        """
        timestamps = data_frame['timestamp'].to_numpy()
        x = timestamps - timestamps[0]  # shift to zero

        # Define new constant sampling points
        num = int(x[-1] * sampling_rate)  # 30 fps
        xx = np.linspace(x[0], x[-1], num)

        frames, features = data_frame.shape
        data = data_frame.to_numpy()

        uniform_sampled_data = []
        for feature in range(features):
            y = data[:, feature]
            f = interpolate.interp1d(x, y, kind="cubic")
            yy = f(xx)
            uniform_sampled_data.append(yy)

        return pd.DataFrame(data=np.array(uniform_sampled_data).T, columns=data_frame.columns)

    def fill_missing_data(self, data, delta=33333):
        _, cols = data.shape
        data_body = data.to_numpy()
        diffs = np.diff(data["timestamp"]) / delta
        diffs = (np.round(diffs) - 1).astype(np.uint32)
        print(f'Number of missing data points: {np.sum(diffs)}')

        inc = 0
        for idx, missing_frames in enumerate(diffs):
            if missing_frames <= 0:
                continue

            for j in range(missing_frames):
                data_body = np.insert(data_body, idx + inc + j + 1, np.full(cols, np.nan), axis=0)

            inc += missing_frames

        data = pd.DataFrame(data_body, columns=data.columns).interpolate(method='quadratic')
        return data

    def multiply_matrix(self, matrix, translation=np.array([0, 0, 0])):
        """
        Multiply all data points with a matrix and add a translation vector
        :param matrix: rotation or coordinate system transformation matrix
        :param translation: translation vector
        :return: None
        """
        data = self.get_data(with_timestamps=False)
        samples, features = data.shape
        result = matrix * data.reshape(-1, 3).T + translation.reshape(3, 1)
        final_result = result.T.reshape(samples, features)

        # check if timestamps in data are present
        if 'timestamp' in self.data:
            timestamps = self.data['timestamp'].to_numpy()
            final_result = np.insert(final_result, 0, timestamps, axis=1)

        self.update_data_body(final_result)

    def update_data_body(self, data):
        """
        Updates a new data body to joints
        :param data: nd-array that contains new data
        :return None
        """
        samples, features = data.shape  # the new data format
        current_columns = self.data.columns  # current columns in data frame

        assert features == len(current_columns), f"Tries to assign data with wrong shape to {self}"
        self.data = pd.DataFrame(data=data, columns=current_columns)

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

    def get_synchronization_signal(self) -> np.ndarray:
        spine_navel = self['spine_navel'].to_numpy()
        return spine_navel[:, 1]  # Only return y-axis

    def get_timestamps(self) -> np.ndarray:
        timestamps = self.data['timestamp'].to_numpy() / 1e6  # convert to seconds
        return timestamps

    def get_synchronization_data(self):
        clock = self.get_timestamps()
        raw_data = normalize_signal(apply_butterworth_filter(self.get_synchronization_signal()))
        acc_data = normalize_signal(np.gradient(np.gradient(raw_data)))  # Calculate 2nd derivative
        peaks = find_peaks(-acc_data, height=self.height, prominence=self.prominence, distance=self.distance)
        return clock, raw_data, acc_data, peaks

    @property
    def sampling_frequency(self):
        return self._sampling_frequency

    def __repr__(self):
        return "azure"
