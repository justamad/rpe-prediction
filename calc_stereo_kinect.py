from rpe_prediction.stereo_cam import StereoAzure
from rpe_prediction.devices import AzureKinect
from rpe_prediction.rendering import SkeletonViewer
from rpe_prediction.scripts.calibration import calculate_calibration
from rpe_prediction.processing import apply_butterworth_filter_dataframe
from os.path import join

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
excluded_joints = ["eye", "ear", "nose", "wrist", "hand", "thumb"]


path = "data/raw/CCB8AD"
cali = "data/raw/AEBA3A"

# Calculate external rotation
rot, trans = calculate_calibration(join(cali, "calibration"), show=False)
cam = StereoAzure(join(path, "azure", "05_master"), join(path, "azure", "05_sub"))
cam.apply_external_rotation(rot, trans)
# cam.calculate_spatial_on_data()

joints = AzureKinect.get_skeleton_connections(cam.master.position_data)

mas_data = cam.mas_position
sub_data = cam.sub_position
avg = cam.fuse_sub_and_master_cameras(alpha=0.1, window_size=9)
f_avg = apply_butterworth_filter_dataframe(avg, sampling_frequency=30, order=4, fc=6)

# norm_avg = 0.5 * sub_data + 0.5 * mas_data
# norm_avg = apply_butterworth_filter_dataframe(norm_avg, sampling_frequency=30, order=4, fc=6)

joint = 'pelvis (x) '

plt.plot(mas_data[joint], label="MAS Ankle Left (Y)")
plt.plot(sub_data[joint], label="SUB Ankle Left (Y)")
plt.plot(avg[joint], label="Average Ankle Left (Y)")
plt.plot(f_avg[joint], label="Normal Average Ankle Left (Y)")
plt.legend()
plt.show()

# viewer = SkeletonViewer(sphere_radius=0.005)
# viewer.add_skeleton(norm_avg, joints)
# viewer.add_skeleton(sub_data, joints)
#viewer.add_skeleton(avg, joints)
# viewer.show_window()
