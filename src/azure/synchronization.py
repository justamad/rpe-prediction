from src.azure import AzureKinect

import numpy as np
import matplotlib.pyplot as plt


def synchronize_azures(master_cam, subord_cam, delay=16000):
    master_data = master_cam.get_data(with_timestamps=True)
    subord_data = subord_cam.get_data(with_timestamps=True)

    # Subtract delay from subordinate device
    master_timestamps = master_data[:, 0]
    subord_timestamps = subord_data[:, 0] - delay
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

    return master_data[:, 1:], subord_data[:, 1:]
