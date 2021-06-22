from rpe_prediction.devices import AzureKinect
from .icp import find_rigid_transformation_svd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class StereoAzure(object):

    def __init__(self, master_path, sub_path, delay=0.001):
        # Read in master device
        self.master = AzureKinect(master_path)
        self.master.process_raw_data()

        # Read in sub device
        self.sub = AzureKinect(sub_path)
        self.sub.process_raw_data()

        self.delay = delay
        self.synchronize_temporal()

    def synchronize_temporal(self):
        """
        Synchronize the two camera stream temporally
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

    def fuse_cameras(self, alpha, window_size=5, show=False, path=None, joint='pelvis (y) '):
        """
        Calculate the fusion of sub and master cameras. Data should be calibrated as good as possible
        @param alpha: coefficient for dominant skeleton side
        @param window_size: a window size of gradient averages
        @param show: Flag whether results should be plotted
        @param path: Path if result should be plotted
        @param joint: joint name that should be plotted
        @return: Fused skeleton data in a pandas array
        """
        df_sub = self.sub_position.reset_index(drop=True)
        df_master = self.mas_position.reset_index(drop=True)

        grad_a = pd.DataFrame(np.square(np.gradient(df_sub.to_numpy(), axis=0)), columns=df_sub.columns)
        grad_b = pd.DataFrame(np.square(np.gradient(df_master.to_numpy(), axis=0)), columns=df_master.columns)
        grad_a = grad_a.rolling(window=window_size, min_periods=1, center=True).mean()
        grad_b = grad_b.rolling(window=window_size, min_periods=1, center=True).mean()
        weights_sub, weights_master = self.calculate_percentage(grad_a, grad_b)

        alpha_weights_sub = pd.DataFrame(np.zeros(df_sub.shape), columns=df_sub.columns)
        alpha_weights_sub[alpha_weights_sub.filter(like='left').columns] += alpha
        alpha_weights_sub[alpha_weights_sub.filter(like='right').columns] -= alpha

        alpha_weights_master = pd.DataFrame(np.zeros(df_sub.shape), columns=df_sub.columns)
        alpha_weights_master[alpha_weights_master.filter(like='right').columns] += alpha
        alpha_weights_master[alpha_weights_master.filter(like='left').columns] -= alpha

        weight_sub_nd = (1 - weights_sub + alpha_weights_sub)
        weight_sub_nd[weight_sub_nd > 1] = 1
        weight_sub_nd[weight_sub_nd < 0] = 0

        weight_master_nd = (1 - weights_master + alpha_weights_master)
        weight_master_nd[weight_master_nd > 1] = 1
        weight_master_nd[weight_master_nd < 0] = 0

        weight_sub = pd.DataFrame(weight_sub_nd, columns=df_sub.columns)
        weight_master = pd.DataFrame(weight_master_nd, columns=df_master.columns)
        fused_skeleton = weight_sub * df_sub + weight_master * df_master

        if show:
            moving_average = (df_sub + df_master) / 2  # Regular moving average for comparison
            plt.plot(df_sub[joint], label="Left Camera")
            plt.plot(df_master[joint], label="Right Camera")
            plt.plot(fused_skeleton[joint], label="Fusion Approach")
            plt.plot(moving_average[joint], label="Moving Average")
            # plt.plot(weight_sub[joint], label="Weights Sub")
            # plt.plot(weight_master[joint], label="Weights Master")
            plt.legend()
            plt.xlabel("Frames (30Hz)")
            plt.ylabel("Distance (mm)")
            plt.title(f"{joint.title().replace('_', ' ')}")
            plt.tight_layout()

            if path is not None:
                plt.savefig(path)
            else:
                plt.show()

            plt.close()
            plt.clf()
            plt.cla()

        return fused_skeleton

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

    @staticmethod
    def calculate_percentage(grad_a, grad_b):
        rows, cols = grad_a.shape
        gradient_stack = np.stack([grad_a, grad_b], axis=2)
        sums = np.sum(gradient_stack, axis=2).reshape((rows, cols, 1))
        weights = gradient_stack / sums
        weights_a = pd.DataFrame(weights[:, :, 0], columns=grad_a.columns)
        weights_b = pd.DataFrame(weights[:, :, 1], columns=grad_b.columns)
        return weights_a, weights_b

    def cut_skeleton_data(self, start_index: int, end_index: int):
        """
        Cut both cameras based on given start and end index
        @param start_index: start index for cutting
        @param end_index: end index for cutting
        @return: None
        """
        self.sub.cut_data_by_index(start_index, end_index)
        self.master.cut_data_by_index(start_index, end_index)

    def reduce_skeleton_joints(self):
        self.sub.remove_unnecessary_joints()
        self.master.remove_unnecessary_joints()

    @property
    def sub_position(self):
        return self.sub.position_data

    @property
    def mas_position(self):
        return self.master.position_data
