from src.devices import MultiAzure
from src.rendering import SkeletonViewer
from os.path import join

import numpy as np

path = "data/raw/1E2DB6/azure"

rot = np.loadtxt("rot.np")
trans = np.loadtxt("trans.np").reshape(3, 1)

print(rot.shape)
print(trans.shape)

cam = MultiAzure(join(path, "01_master"), join(path, "01_sub"), None, None)
cam.external_rotation(rot, trans)

viewer = SkeletonViewer()
viewer.add_skeleton(cam.master.position_data.to_numpy() / 1000)
viewer.add_skeleton(cam.sub.position_data.to_numpy() / 1000)
viewer.show_window()
