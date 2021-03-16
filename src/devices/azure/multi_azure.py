from .azure import AzureKinect
from src.devices.azure.calibration import read_calibration_folder, find_rigid_transformation_svd

import numpy as np
import matplotlib
matplotlib.use("TkAgg")


class MultiAzure(AzureKinect):

    def __init__(self, master_path, sub_path, calibration_path, delay=1000):
        master = AzureKinect(master_path)
        sub = AzureKinect(sub_path)
        self.delay = delay

        # Temporal alignment
        master, sub = self.synchronize_cameras(master, sub)

        # Spatial alignment
        points_a, points_b = read_calibration_folder(calibration_path)
        rotation, translation = find_rigid_transformation_svd(points_a, points_b)
        master.multiply_matrix(rotation, translation)

        super().__init__(master)

    def synchronize_cameras(self, master, sub):
        master_data = master.get_data(with_timestamps=True)
        subord_data = sub.get_data(with_timestamps=True)

        # Subtract delay from subordinate device
        master_timestamps = master_data[:, 0]
        subord_timestamps = subord_data[:, 0]
        print(list(np.diff(master_timestamps)))

        first_mutual_frame = max(master_timestamps[0], subord_timestamps[0])
        print(first_mutual_frame)

        # Synchronize data in the beginning
        master_begin = np.argmin(np.abs(master_timestamps - first_mutual_frame))
        subord_begin = np.argmin(np.abs(subord_timestamps - first_mutual_frame))

        master_data = master_data[master_begin:, :]
        subord_data = subord_data[subord_begin:, :]

        # Cut end
        min_length = min(len(master_data), len(subord_data))
        master_data = master_data[:min_length, :]
        subord_data = subord_data[:min_length, :]

        t1 = master_data[:, 0]
        t2 = subord_data[:, 0]
        print(list(t1 - t2))
        # master.update_data_body(master_data)
        # sub.update_data_body(subord_data)
        return master, sub
