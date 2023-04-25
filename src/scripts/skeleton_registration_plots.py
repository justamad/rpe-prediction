from src.camera import StereoAzure
from os.path import join
from PyMoCapViewer import MoCapViewer

import plot_settings as ps
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt

import os

master_path = "../../../../INTENSO/RPE_Data/9AE368/azure/01_master/positions_3d.csv"
sub_path = "../../../../INTENSO/RPE_Data/9AE368/azure/01_sub/positions_3d.csv"

azure = StereoAzure(master_path, sub_path)

master_df = azure.mas_position
sub_df = azure.sub_position
fused_df = azure._fused

plt.figure(figsize=(ps.image_width * 2 * ps.cm, ps.image_width * ps.cm), dpi=300)

plt.plot(master_df["KNEE_RIGHT (x)"], label="Master")
plt.plot(sub_df["KNEE_RIGHT (x)"], label="Sub")
plt.plot(fused_df["KNEE_RIGHT (x)"], label="Fused")
plt.legend()
plt.xlabel("Time (Seconds)")
plt.ylabel("Distance (Meters)")
plt.tight_layout()
plt.savefig("knee_right.pdf", dpi=300)
# plt.show()


viewer = MoCapViewer(sphere_radius=0.010, bg_color="white", grid_axis=None, sampling_frequency=30)
viewer.add_skeleton(azure.mas_position, skeleton_connection="azure", color="red")
viewer.add_skeleton(azure.sub_position, skeleton_connection="azure", color="green")
viewer.add_skeleton(azure._fused, skeleton_connection="azure", color="blue")
viewer.show_window()
