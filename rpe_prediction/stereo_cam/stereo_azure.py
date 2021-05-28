from rpe_prediction.devices import AzureKinect
from .icp import find_rigid_transformation_svd

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

    def apply_external_rotation(self, rotation, translation):
        """
        Apply an external affine transformation consisting of rotation and translation
        @param rotation: A 3x3 rotation matrix
        @param translation: A 1x3 translation vector
        """
        self.sub.multiply_matrix(rotation, translation)

    def calculate_spatial_on_data(self, show=False):
        master_position = self.master.position_data.to_numpy()  # [400:600, :]
        sub_position = self.sub.position_data.to_numpy()  # [400:600, :]

        # Find the best affine transformation
        rotation, translation = find_rigid_transformation_svd(master_position.reshape(-1, 3),
                                                              sub_position.reshape(-1, 3),
                                                              show=show)

        trans_init = np.eye(4)
        trans_init[0:3, 0:3] = rotation
        trans_init[0:3, 3] = translation.reshape(3)
        self.master.multiply_matrix(rotation, translation)

    def calculate_fusion(self, alpha, window_size=5):
        """
        TODO: Implement show argument for plotting the agreement instead of returning the weights
        Calculate the fusion of sub and master cameras. Data should be calibrated as good as possible
        @param alpha: coefficient for dominant skeleton side
        @param window_size: a window size of gradient averages
        @return: Fused skeleton data in a pandas array
        """
        df_sub = self.sub_position
        df_master = self.mas_position

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
        return fused_skeleton

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

    @property
    def sub_position(self):
        return self.sub.position_data

    @property
    def mas_position(self):
        return self.master.position_data

# def draw_registration_result(source, target, transformation):
#     source_temp = copy.deepcopy(source)
#     target_temp = copy.deepcopy(target)
#     # source_temp.paint_uniform_color([1, 0.706, 0])
#     # target_temp.paint_uniform_color([0, 0.651, 0.929])
#     source_temp.transform(transformation)
#     o3d.visualization.draw_geometries([source_temp, target_temp], mesh_show_back_face=False)
