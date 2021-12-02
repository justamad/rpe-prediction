from src.processing import apply_butterworth_filter
from src.camera import AzureKinect
from src.camera.utils.icp import find_rigid_transformation_svd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging


class StereoAzure(object):

    def __init__(
            self,
            master_path: str,
            sub_path: str,
            delay: float = 0.001,
    ):
        self._master = AzureKinect(master_path)
        self._sub = AzureKinect(sub_path)

        self._delay = delay
        self._synchronize_temporal()

    def _synchronize_temporal(self):
        self._sub.add_delta_to_timestamps(-self._delay)
        start_point = self._master.timestamps[0]
        self._master.add_delta_to_timestamps(-start_point)
        self._sub.add_delta_to_timestamps(-start_point)

        # Cut data based on same timestamps
        minimum = int(np.argmin(np.abs(self._sub.timestamps)))
        length = min(len(self._master.timestamps), len(self._sub.timestamps) - minimum)
        self._master.cut_data_by_index(0, length)
        self._sub.cut_data_by_index(minimum, minimum + length)
        self._master.set_new_timestamps(self._sub.timestamps)

    def apply_external_rotation(
            self,
            rotation: np.ndarray,
            translation: np.ndarray,
    ):
        self._sub.apply_affine_transformation(rotation, translation)

    def fuse_cameras(
            self,
            show: bool = False,
            pp=None,
    ):
        df_sub = self.sub_position.reset_index(drop=True)
        df_master = self.mas_position.reset_index(drop=True)

        m_err_1, s_err_1 = self._calculate_error_between_both_cameras()
        self._calculate_affine_transformation_based_on_data()
        m_err_2, s_err_2 = self._calculate_error_between_both_cameras()

        logging.info(f"Joint errors: m={m_err_1:.2f}, s={s_err_2:.2f} mm, after: m={m_err_2:.2f}, s={s_err_2:.2f}")

        # Variance average
        var_sub = df_sub.var()
        var_mas = df_master.var()
        averaged = (var_sub * df_master + var_mas * df_sub) / (var_sub + var_mas)
        average_filtered = apply_butterworth_filter(
            df=averaged,
            cutoff=4,
            sampling_rate=30,
            order=4,
        )

        if show:
            plt.close()
            plt.figure()

            for joint in df_sub.columns:
                plt.plot(df_sub[joint], color="red", label="Left Sensor")
                plt.plot(df_master[joint], color="blue", label="Right Sensor")
                plt.plot(average_filtered[joint], label="Butterworth 4 Hz")
                plt.xlabel("Frames [1/30 s]")
                plt.ylabel("Distance [mm]")
                plt.title(f"{joint.title().replace('_', ' ')}")
                plt.legend()
                plt.tight_layout()
                pp.save_figure()
                plt.clf()

        average_filtered = average_filtered.set_index(self.sub_position.index)
        average_filtered.index = pd.to_datetime(average_filtered.index, unit="s")
        return average_filtered

    def _calculate_affine_transformation_based_on_data(
            self,
            show: bool = False,
    ):
        rotation, translation = find_rigid_transformation_svd(
            self._master.data.to_numpy().reshape(-1, 3),
            self._sub.data.to_numpy().reshape(-1, 3),
            show=show
        )
        self._master.apply_affine_transformation(rotation, translation)

    def _calculate_error_between_both_cameras(self):
        diff = (self._sub.data.to_numpy() - self._master.data.to_numpy()) ** 2
        distances = np.sqrt(diff.reshape(-1, 3).sum(axis=1))
        return np.mean(distances), np.std(distances)

    def cut_skeleton_data(
            self,
            start_index: int,
            end_index: int
    ):
        self._sub.cut_data_by_label(start_index, end_index)
        self._master.cut_data_by_label(start_index, end_index)

    def reduce_skeleton_joints(self):
        self._sub.remove_unnecessary_joints()
        self._master.remove_unnecessary_joints()

    @property
    def sub_position(self):
        return self._sub.data

    @property
    def mas_position(self):
        return self._master.data
