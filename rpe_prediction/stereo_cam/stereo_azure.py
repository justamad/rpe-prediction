from rpe_prediction.devices import AzureKinect
from rpe_prediction.processing import apply_butterworth_filter_dataframe
from .icp import find_rigid_transformation_svd

import numpy as np
import pandas as pd
import open3d as o3d
import copy
import matplotlib.pyplot as plt


class StereoAzure(object):

    def __init__(self, master_path, sub_path, delay=0.001):
        # Read in master device
        self.master = AzureKinect(master_path)
        self.master.process_raw_data()

        # Read in sub device
        self.sub = AzureKinect(sub_path)
        self.sub.process_raw_data()

        self.delay = delay
        self.time_synchronization()

        self.sub_pos = self.sub.position_data
        self.mas_pos = self.master.position_data

    def time_synchronization(self):
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
        self.sub.multiply_matrix(rotation, translation)
        self.sub_pos = self.sub.position_data
        self.mas_pos = self.master.position_data

    def plot_axis(self):
        row, cols = self.sub_pos.shape

        for i in range(cols):
            column_name = self.sub_pos.columns[i]
            sub_data = self.sub_pos[column_name]
            mas_data = self.mas_pos[column_name]
            diff = np.abs(sub_data - mas_data)
            avg = self.avg_df[column_name]

            plt.plot(sub_data, label="sub")
            plt.plot(mas_data, label="master")
            plt.plot(diff, label="diff")
            plt.plot(avg, label="Average")
            plt.ylabel("MM")
            plt.xlabel("Frames")
            plt.title(f"{column_name} - Diff: {np.mean(diff)}, {np.std(diff)}")
            plt.legend()
            plt.show()

    def spatial_alignment(self):
        master_position = self.master.position_data.to_numpy()[1500:1600, :]
        sub_position = self.sub.position_data.to_numpy()[1500:1600, :]

        # Spatial alignment
        rotation, translation = find_rigid_transformation_svd(master_position.reshape(-1, 3),
                                                              sub_position.reshape(-1, 3), True)

        trans_init = np.eye(4)
        trans_init[0:3, 0:3] = rotation
        trans_init[0:3, 3] = translation.reshape(3)
        self.master.multiply_matrix(rotation, translation)

        # Visualize Point clouds
        # pcd_m = o3d.io.read_point_cloud(self.point_cloud_master)
        # pcd_m.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # pcd_s = o3d.io.read_point_cloud(self.point_cloud_sub)
        # pcd_s.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # pcd_s.transform(affine)
        # draw_registration_result(pcd_s, pcd_m, trans_init)

        print("Apply point-to-plane ICP")
        # threshold = 0.02
        # reg_p2l = o3d.pipelines.registration.registration_icp(
        #     pcd_s, pcd_m, threshold, trans_init,
        #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
        # print(reg_p2l)
        # diff = reg_p2l.transformation - trans_init
        # print(f"Diff {diff}")
        # draw_registration_result(pcd_s, pcd_m, reg_p2l.transformation)


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], mesh_show_back_face=False)
