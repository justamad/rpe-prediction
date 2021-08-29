from rpe_prediction.processing import apply_butterworth_filter
from rpe_prediction.devices import AzureKinect
from .icp import find_rigid_transformation_svd

import matplotlib.pyplot as plt
import numpy as np


class StereoAzure(object):

    def __init__(self, master_path, sub_path, delay: float = 0.001):
        """
        Constructor for Azure Kinect stereo camera
        @param master_path: path to master camera
        @param sub_path: path to subordinate camera
        @param delay: the IR-camera offset between both cameras with respect to the master device
        """
        # Read in master and subordinate devices
        self.master = AzureKinect(master_path)
        self.sub = AzureKinect(sub_path)

        self.delay = delay
        self.synchronize_temporal()

    def synchronize_temporal(self):
        """
        Synchronize the two camera streams temporally
        @return: None
        """
        self.sub.shift_clock(-self.delay)
        start_point = self.master.timestamps[0]
        self.master.shift_clock(-start_point)
        self.sub.shift_clock(-start_point)

        # Cut data based on same timestamps
        minimum = int(np.argmin(np.abs(self.sub.timestamps)))
        length = min(len(self.master.timestamps), len(self.sub.timestamps) - minimum)
        self.master.cut_data_by_index(0, length)
        self.sub.cut_data_by_index(minimum, minimum + length)
        self.master.set_timestamps(self.sub.timestamps)

    def apply_external_rotation(self, rotation, translation):
        """
        Apply an external affine transformation consisting of rotation and translation
        @param rotation: A 3x3 rotation matrix
        @param translation: A 1x3 translation vector
        """
        self.sub.multiply_matrix(rotation, translation)

    def calculate_affine_transform_based_on_data(self, show=False):
        """
        Calculate an affine transformation to register one skeleton to the other
        @param show: Flag if the calibration result should be shown
        @return: None
        """
        rotation, translation = find_rigid_transformation_svd(self.master.data.to_numpy().reshape(-1, 3),
                                                              self.sub.data.to_numpy().reshape(-1, 3),
                                                              show=show)

        self.master.multiply_matrix(rotation, translation)

    def fuse_cameras(self, show=False, pp=None):
        """
        Fusing both Kinect data frames into a single one using a moving-average filter
        @param show: flag whether data shuld be shown
        @param pp:
        @return:
        """
        df_sub = self.sub_position.reset_index(drop=True)
        df_master = self.mas_position.reset_index(drop=True)

        # Variance average
        var_sub = df_sub.var()
        var_mas = df_master.var()
        average_var = (var_sub * df_master + var_mas * df_sub) / (var_sub + var_mas)
        average_f4 = apply_butterworth_filter(average_var, cutoff=4, sampling_rate=30, order=4)

        if show:
            plt.close()
            plt.figure()

            for joint in df_sub.columns:
                plt.plot(df_sub[joint], color="red", label="Left Sensor")
                plt.plot(df_master[joint], color="blue", label="Right Sensor")
                plt.plot(average_f4[joint], label="Butterworth 4 Hz")
                plt.xlabel("Frames [1/30 s]")
                plt.ylabel("Distance [mm]")
                plt.title(f"{joint.title().replace('_', ' ')}")
                plt.legend()
                plt.tight_layout()
                pp.save_figure()
                plt.clf()

        return average_f4.set_index(self.sub_position.index)

    def check_agreement_of_both_cameras(self):
        """
        Calculate the euclidean distance between left and right camera
        :return: Tuple of of (mean, std) of euclidean distance
        """
        diff = (self.sub.data.to_numpy() - self.master.data.to_numpy()) ** 2
        distances = np.sqrt(diff.reshape(-1, 3).sum(axis=1))
        return np.mean(distances), np.std(distances)

    def cut_skeleton_data(self, start_index: int, end_index: int):
        """
        Cut both cameras based on given start and end index
        @param start_index: start index for cutting
        @param end_index: end index for cutting
        @return: None
        """
        self.sub.cut_data_by_label(start_index, end_index)
        self.master.cut_data_by_label(start_index, end_index)

    def reduce_skeleton_joints(self):
        self.sub.remove_unnecessary_joints()
        self.master.remove_unnecessary_joints()

    @property
    def sub_position(self):
        return self.sub.data

    @property
    def mas_position(self):
        return self.master.data
