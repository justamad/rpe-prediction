from rpe_prediction.stereo_cam import StereoAzure
from rpe_prediction.devices import AzureKinect
from rpe_prediction.rendering import SkeletonViewer
from calibration import calculate_calibration
from rpe_prediction.processing import apply_butterworth_filter_dataframe
from os.path import join

import pandas as pd
import numpy as np

path = "data/raw/AEBA3A"

# Calculate external rotation
rot, trans = calculate_calibration(join(path, "calibration"), show=False)

cam = StereoAzure(join(path, "azure", "01_master"), join(path, "azure", "01_sub"))
cam.apply_external_rotation(rot, trans)

joints = AzureKinect.get_skeleton_connections(cam.master.position_data)


def calculate_percentage_df(grad_a, grad_b):
    shape = grad_a.shape
    mat = np.stack([grad_a, grad_b], axis=2)
    sums = np.sum(mat, axis=2).reshape((shape[0], shape[1], 1))
    mat = mat / sums
    return 1 - mat


def calculate_fusion(df_a, df_b, alpha, beta, window_size=5):
    grad_a = np.abs(np.gradient(df_a.to_numpy(), axis=0))
    grad_b = np.abs(np.gradient(df_b.to_numpy(), axis=0))
    grad_a = pd.DataFrame(grad_a).rolling(window=window_size, min_periods=1, center=True).mean()
    grad_b = pd.DataFrame(grad_b).rolling(window=window_size, min_periods=1, center=True).mean()
    percentage = calculate_percentage_df(grad_a, grad_b)
    new_mat = alpha * percentage[:, :, 0] * df_a + beta * percentage[:, :, 1] * df_b
    return new_mat


mas_data = cam.mas_pos
sub_data = cam.sub_pos
avg = calculate_fusion(sub_data, mas_data, 1.02, 0.98, window_size=9)
avg = apply_butterworth_filter_dataframe(avg, sampling_frequency=30, order=4, fc=6)

viewer = SkeletonViewer()
viewer.add_skeleton(mas_data.to_numpy() / 1000, joints)
viewer.add_skeleton(sub_data.to_numpy() / 1000, joints)
viewer.add_skeleton(avg.to_numpy() / 1000, joints)
viewer.show_window()
