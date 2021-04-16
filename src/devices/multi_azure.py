from src.devices import AzureKinect
from processing import find_rigid_transformation_svd
from src.rendering import SkeletonViewer

import numpy as np
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

        # window = SkeletonViewer()
        # window.add_skeleton(master_position)
        # window.add_skeleton(sub_position)
        # window.show_window()

        # Spatial alignment
        rotation, translation = find_rigid_transformation_svd(master_position.reshape(-1, 3),
                                                              sub_position.reshape(-1, 3), True)
        print(rotation)
        # master.multiply_matrix(rotation, translation)


if __name__ == '__main__':
    cam = MultiAzure("../../data/justin/azure/01_master", "../../data/justin/azure/01_sub",
                     "../../data/3001188.ply", "../../data/30.ply")
