from rpe_prediction.processing import apply_butterworth_filter
from rpe_prediction.camera import AzureKinect
from .icp import find_rigid_transformation_svd

import matplotlib.pyplot as plt
import numpy as np


class StereoAzure(object):

    def __init__(self, master_path: str, sub_path: str, delay: float = 0.001):
        self.master = AzureKinect(master_path)
        self.sub = AzureKinect(sub_path)

        self.delay = delay
        self.synchronize_temporal()

    def synchronize_temporal(self):
        self.sub.add_delta_to_timestamps(-self.delay)
        start_point = self.master.timestamps[0]
        self.master.add_delta_to_timestamps(-start_point)
        self.sub.add_delta_to_timestamps(-start_point)

        # Cut data based on same timestamps
        minimum = int(np.argmin(np.abs(self.sub.timestamps)))
        length = min(len(self.master.timestamps), len(self.sub.timestamps) - minimum)
        self.master.cut_data_by_index(0, length)
        self.sub.cut_data_by_index(minimum, minimum + length)
        self.master.set_new_timestamps(self.sub.timestamps)

    def apply_external_rotation(self, rotation: np.ndarray, translation: np.ndarray):
        self.sub.apply_affine_transformation(rotation, translation)

    def calculate_affine_transform_based_on_data(self, show: bool = False):
        rotation, translation = find_rigid_transformation_svd(self.master.data.to_numpy().reshape(-1, 3),
                                                              self.sub.data.to_numpy().reshape(-1, 3),
                                                              show=show)

        self.master.apply_affine_transformation(rotation, translation)

    def fuse_cameras(self, show: bool = False, pp=None):
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

    def calculate_error_between_both_cameras(self):
        diff = (self.sub.data.to_numpy() - self.master.data.to_numpy()) ** 2
        distances = np.sqrt(diff.reshape(-1, 3).sum(axis=1))
        return np.mean(distances), np.std(distances)

    def cut_skeleton_data(self, start_index: int, end_index: int):
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
