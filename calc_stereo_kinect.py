from rpe_prediction.stereo_cam import StereoAzure
from rpe_prediction.devices import AzureKinect
from rpe_prediction.rendering import SkeletonViewer
from calibration import calculate_calibration
from rpe_prediction.processing import apply_butterworth_filter_dataframe
from os.path import join

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

path = "data/raw/AEBA3A"
# Calculate external rotation
rot, trans = calculate_calibration(join(path, "calibration"), show=False)
cam = StereoAzure(join(path, "azure", "01_master"), join(path, "azure", "01_sub"))
cam.apply_external_rotation(rot, trans)
cam.calculate_spatial_on_data()

joints = AzureKinect.get_skeleton_connections(cam.master.position_data)

mas_data = cam.mas_position
sub_data = cam.sub_position
avg = cam.calculate_fusion(alpha=1.1, beta=0.9, window_size=9)
# avg = apply_butterworth_filter_dataframe(avg, sampling_frequency=30, order=4, fc=6)


plt.plot(mas_data['ankle_left (y) '], label="SUB Ankle Left (Y)")
plt.plot(sub_data['ankle_left (y) '], label="SUB Ankle Left (Y)")
plt.plot(avg['ankle_left (y) '], label="SUB Ankle Left (Y)")
plt.legend()
plt.show()

# viewer = SkeletonViewer()
# viewer.add_skeleton(mas_data.to_numpy() / 1000, joints)
# viewer.add_skeleton(sub_data.to_numpy() / 1000, joints)
# viewer.add_skeleton(avg.to_numpy() / 1000, joints)
# viewer.show_window()
