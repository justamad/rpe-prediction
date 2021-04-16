from src.devices import AzureKinect
from processing import find_rigid_transformation_svd

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("TkAgg")


class MultiAzure(object):

    def __init__(self, master_path, sub_path, pointcloud_master, pointcloud_sub, delay=0.001):
        master = AzureKinect(master_path)
        sub = AzureKinect(sub_path)
        master.process_raw_data()
        sub.process_raw_data()

        # Synchronize master and sub devices
        sub.shift_clock(-delay)
        start_point = master.timestamps[0]
        master.shift_clock(-start_point)
        sub.shift_clock(-start_point)

        # Cut data based on same timestamps
        minimum = int(np.argmin(np.abs(sub.timestamps)))
        length = min(len(master.timestamps), len(sub.timestamps) - minimum)
        master.cut_data_by_index(0, length)
        sub.cut_data_by_index(minimum, minimum + length)

        master_position = master.position_data.to_numpy()[1500:1600, :]
        sub_position = sub.position_data.to_numpy()[1500:1600, :]

        # Spatial alignment
        rotation, translation = find_rigid_transformation_svd(master_position.reshape(-1, 3),
                                                              sub_position.reshape(-1, 3), False)

        affine = np.eye(4)
        affine[0:3, 0:3] = rotation
        affine[0:3, 3] = translation.reshape(3)

        # Visualize Point clouds
        pcd_m = o3d.io.read_point_cloud(pointcloud_master)
        pcd_m.remove_statistical_outlier(50, 10)

        pcd_s = o3d.io.read_point_cloud(pointcloud_sub)
        pcd_s.remove_statistical_outlier(50, 10)
        pcd_s.transform(affine)
        o3d.visualization.draw_geometries([pcd_m, pcd_s], mesh_show_back_face=True)


if __name__ == '__main__':
    cam = MultiAzure("../../data/justin/azure/01_master", "../../data/justin/azure/01_sub",
                     "../../data/justin/3001188.ply", "../../data/justin/3000188.ply")
