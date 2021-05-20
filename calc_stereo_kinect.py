from rpe_prediction.stereo_cam import StereoAzure
from rpe_prediction.rendering import SkeletonViewer
from os.path import join

from calibration import calculate_calibration

import numpy as np

path = "data/raw/AEBA3A"

# Find external rotation
rot, trans = calculate_calibration(join(path, "calibration"), show=False)

cam = StereoAzure(join(path, "azure", "01_master"), join(path, "azure", "01_sub"))
cam.apply_external_rotation(rot, trans)

viewer = SkeletonViewer()
viewer.add_skeleton(cam.master.position_data.to_numpy() / 1000)
viewer.add_skeleton(cam.sub.position_data.to_numpy() / 1000)
viewer.show_window()
