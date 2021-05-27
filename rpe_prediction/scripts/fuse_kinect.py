from rpe_prediction.config import SubjectDataIterator, KinectFusionLoaderSet
from rpe_prediction.processing import segment_exercises_based_on_joint
from calibration import calculate_calibration
from rpe_prediction.stereo_cam import StereoAzure
from rpe_prediction.rendering import SkeletonViewer
from os.path import join

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

file_iterator = SubjectDataIterator("../../data/raw", KinectFusionLoaderSet())
example = np.loadtxt("../../exe.np")

path = "../../data/raw/AEBA3A"
rot, trans = calculate_calibration(join(path, "calibration"), show=False)


def fuse_kinect_data(iterator):
    for set_data in iterator:
        sub_path, master_path = set_data['azure']
        print(sub_path)
        azure = StereoAzure(master_path=master_path, sub_path=sub_path)
        azure.apply_external_rotation(rot, trans)
        # avg, _, _ = azure.calculate_fusion(alpha=0.1, window_size=9)
        sub = azure.sub_position['pelvis (y) ']
        segment_exercises_based_on_joint(sub, example, 20, 4, True)
        # plt.plot(sub)
        # plt.plot(avg['pelvis (y) '])
        # plt.show()
        # viewer = SkeletonViewer()
        # viewer.add_skeleton(azure)
        # viewer.show_window()


if __name__ == '__main__':
    fuse_kinect_data(file_iterator.iterate_over_all_subjects())
