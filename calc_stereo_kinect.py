from rpe_prediction.stereo_cam import StereoAzure
from rpe_prediction.devices import AzureKinect
from rpe_prediction.rendering import SkeletonViewer
from calibration import calculate_calibration
from rpe_prediction.processing import apply_butterworth_filter_dataframe
from os.path import join

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

path = "data/raw/AEBA3A"
# Calculate external rotation
rot, trans = calculate_calibration(join(path, "calibration"), show=False)
cam = StereoAzure(join(path, "azure", "01_master"), join(path, "azure", "01_sub"))
cam.apply_external_rotation(rot, trans)
cam.calculate_spatial_on_data()

joints = AzureKinect.get_skeleton_connections(cam.master.position_data)

mas_data = cam.mas_position
sub_data = cam.sub_position
avg = cam.calculate_fusion(alpha=0.1, window_size=9)
# avg = apply_butterworth_filter_dataframe(avg, sampling_frequency=30, order=4, fc=6)

norm_avg = 0.5 * sub_data + 0.5 * mas_data

plt.plot(mas_data['ankle_left (y) '], label="MAS Ankle Left (Y)")
plt.plot(sub_data['ankle_left (y) '], label="SUB Ankle Left (Y)")
plt.plot(avg['ankle_left (y) '], label="Average Ankle Left (Y)")
plt.plot(norm_avg['ankle_left (y) '], label="Normal Average Ankle Left (Y)")
plt.legend()
plt.show()

viewer = SkeletonViewer()
viewer.add_skeleton(mas_data, joints)
viewer.add_skeleton(sub_data, joints)
viewer.add_skeleton(avg, joints)
# viewer.show_window()
