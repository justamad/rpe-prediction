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

    def calculate_spatial_on_data(self):
        master_position = self.master.position_data.to_numpy()  # [400:600, :]
        sub_position = self.sub.position_data.to_numpy()  # [400:600, :]

        # Spatial alignment
        rotation, translation = find_rigid_transformation_svd(master_position.reshape(-1, 3),
                                                              sub_position.reshape(-1, 3), True)

        trans_init = np.eye(4)
        trans_init[0:3, 0:3] = rotation
        trans_init[0:3, 3] = translation.reshape(3)
        self.master.multiply_matrix(rotation, translation)

    @staticmethod
    def calculate_percentage_df(grad_a, grad_b):
        shape = grad_a.shape
        mat = np.stack([grad_a, grad_b], axis=2)
        sums = np.sum(mat, axis=2).reshape((shape[0], shape[1], 1))
        mat = mat / sums
        return 1 - mat

    def calculate_fusion(self, alpha, beta, window_size=5):
        """
        Calculate the fusion of sub and master cameras. Data should be calibrated as good as possible
        @param alpha: coefficient for dominant skeleton side
        @param beta: coefficient for weaker skeleton side
        @param window_size: a window size of gradient averages
        @return: Fused skeleton data in a pandas array
        """
        df_sub = self.sub_position
        df_master = self.mas_position

        grad_a = np.square(np.gradient(self.sub_position.to_numpy(), axis=0))
        grad_b = np.square(np.gradient(self.mas_position.to_numpy(), axis=0))
        grad_a = pd.DataFrame(grad_a).rolling(window=window_size, min_periods=1, center=True).mean()
        grad_b = pd.DataFrame(grad_b).rolling(window=window_size, min_periods=1, center=True).mean()
        gradient_weights = self.calculate_percentage_df(grad_a, grad_b)
        fused_skeleton = alpha * gradient_weights[:, :, 0] * df_sub + beta * gradient_weights[:, :, 1] * df_master
        return fused_skeleton

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
