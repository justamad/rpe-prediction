from rpe_prediction.processing import butterworth_filter
from rpe_prediction.devices import AzureKinect
from .icp import find_rigid_transformation_svd

import matplotlib.pyplot as plt
import numpy as np


class StereoAzure(object):

    def __init__(self, master_path, sub_path, delay: float = 0.001):
        """
        Constructor for Azure Kinect stereo camera
        @param master_path:
        @param sub_path:
        @param delay:
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
        # Synchronize master and sub devices
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
        @param show: flag if the result should be shown
        @return: None
        """
        points_a = self.master.position_data.to_numpy()
        points_b = self.sub.position_data.to_numpy()

        gradients_master = np.sum(np.abs(np.gradient(points_a, axis=0)).reshape(-1, 3), axis=1)
        gradients_sub = np.sum(np.abs(np.gradient(points_b, axis=0)).reshape(-1, 3), axis=1)
        weights_total = 1 / (gradients_master + gradients_sub)

        # Find the best affine transformation
        rotation, translation = find_rigid_transformation_svd(points_a.reshape(-1, 3),
                                                              points_b.reshape(-1, 3),
                                                              weights_total.reshape(-1, 1),
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
        average_f4 = butterworth_filter(average_var, fc=4, fs=30, order=4)

        if show:
            plt.close()
            plt.figure()
            plt.xlabel("Frames (30Hz)")
            plt.ylabel("Distance (mm)")

            for joint in df_sub.columns:
                plt.plot(df_sub[joint], color="red", label="Left Sensor")
                plt.plot(df_master[joint], color="blue", label="Right Sensor")
                plt.plot(average_f4[joint], label="Butterworth 4 Hz")
                plt.title(f"{joint.title().replace('_', ' ')}")
                plt.legend()
                plt.tight_layout()
                pp.save_figure()
                plt.clf()

        return average_f4.set_index(self.sub_position.index)

    def check_agreement_of_both_cameras(self, show):
        """
        Calculate the agreement between sub and master camera
        :return: TODO: Calculate Euclidean distance of joints instead of separate axes
        """
        data_a = self.sub.position_data
        data_b = self.master.position_data
        differences = (data_a - data_b).abs().mean(axis=0)
        if show:
            differences.plot.bar(x="joints", y="error", rot=90)

        return differences.mean()

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
        return self.sub.position_data

    @property
    def mas_position(self):
        return self.master.position_data
