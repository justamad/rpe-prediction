from rpe_prediction.config import SubjectDataIterator, KinectFusionLoaderSet
from rpe_prediction.processing import segment_1d_joint_on_example
from calibration import calculate_calibration
from rpe_prediction.stereo_cam import StereoAzure
from rpe_prediction.rendering import SkeletonViewer
from os.path import join

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

file_iterator = SubjectDataIterator("../../data/raw", KinectFusionLoaderSet())
# example = np.loadtxt("../../exe.np")
example = np.loadtxt("example.np")

path = "../../data/raw/AEBA3A"
rot, trans = calculate_calibration(join(path, "calibration"), show=False)


def fuse_kinect_data(iterator):
    for set_data in iterator:
        sub_path, master_path = set_data['azure']
        print(sub_path)
        azure = StereoAzure(master_path=master_path, sub_path=sub_path)
        azure.apply_external_rotation(rot, trans)
        sub = azure.sub_position['pelvis (y) ']
        segment_1d_joint_on_example(sub, example, min_duration=20, std_dev_percentage=0.5, show=True)
        # avg, _, _ = azure.calculate_fusion(alpha=0.1, window_size=9)
        # plt.plot(sub)
        # plt.plot(avg['pelvis (y) '])
        # plt.show()
        # viewer = SkeletonViewer()
        # viewer.add_skeleton(azure)
        # viewer.show_window()


if __name__ == '__main__':
    fuse_kinect_data(file_iterator.iterate_over_all_subjects())
