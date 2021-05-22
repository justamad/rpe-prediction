from rpe_prediction.stereo_cam import StereoAzure
from rpe_prediction.devices import AzureKinect
from rpe_prediction.rendering import SkeletonViewer
from os.path import join

from calibration import calculate_calibration

import numpy as np
import matplotlib
matplotlib.use("TkAgg")

path = "data/raw/AEBA3A"

# Calculate external rotation
rot, trans = calculate_calibration(join(path, "calibration"), show=False)

cam = StereoAzure(join(path, "azure", "01_master"), join(path, "azure", "01_sub"))
cam.apply_external_rotation(rot, trans)
# cam.plot_axis()

joints = AzureKinect.get_skeleton_connections(cam.master.position_data)

mas_data = cam.master.position_data.to_numpy() / 1000
sub_data = cam.sub.position_data.to_numpy() / 1000
avg = (mas_data + sub_data) / 2

viewer = SkeletonViewer()
viewer.add_skeleton(mas_data, joints)
viewer.add_skeleton(sub_data, joints)
# viewer.add_skeleton(avg, joints)
viewer.show_window()
