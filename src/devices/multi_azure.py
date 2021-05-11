from src.devices import AzureKinect
from src.rendering import SkeletonViewer
from .processing import find_rigid_transformation_svd

import numpy as np
import open3d as o3d
import matplotlib
import copy

matplotlib.use("TkAgg")


class MultiAzure(object):

    def __init__(self, master_path, sub_path, point_cloud_master, point_cloud_sub, delay=0.001):
        self.master = AzureKinect(master_path)
        self.sub = AzureKinect(sub_path)
        self.master.process_raw_data()
        self.sub.process_raw_data()
        self.point_cloud_master = point_cloud_master
        self.point_cloud_sub = point_cloud_sub
        self.delay = delay

        self.time_synchronization()
        self.spatial_alignment()

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


if __name__ == '__main__':
    cam = MultiAzure("../../data/justin/azure/01_master", "../../data/justin/azure/01_sub",
                     "../../data/justin/3001188.ply", "../../data/justin/3000188.ply")

    viewer = SkeletonViewer()
    viewer.add_skeleton(cam.master.position_data.to_numpy() / 1000)
    viewer.add_skeleton(cam.sub.position_data.to_numpy() / 1000)
    viewer.show_window()
