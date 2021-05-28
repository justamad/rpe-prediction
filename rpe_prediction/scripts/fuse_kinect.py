from calibration import calculate_calibration
from rpe_prediction.config import SubjectDataIterator, KinectFusionLoaderSet
from rpe_prediction.processing import segment_1d_joint_on_example
from rpe_prediction.stereo_cam import StereoAzure
from rpe_prediction.rendering import SkeletonViewer
from os.path import join

import numpy as np
import matplotlib

matplotlib.use("TkAgg")

file_iterator = SubjectDataIterator("../../data/raw", KinectFusionLoaderSet())
example = np.loadtxt("example.np")

path = "../../data/raw/AEBA3A"
rot, trans = calculate_calibration(join(path, "calibration"), show=False)


def fuse_kinect_data(iterator):
    for set_data in iterator:
        sub_path, master_path = set_data['azure']
        azure = StereoAzure(master_path=master_path, sub_path=sub_path)
        print(f"Agreement initial: {azure.check_agreement_of_both_cameras()}")

        azure.apply_external_rotation(rot, trans)
        print(f"Agreement external: {azure.check_agreement_of_both_cameras()}")

        sub = azure.sub_position['pelvis (y) ']
        repetitions = segment_1d_joint_on_example(sub, example, min_duration=20, std_dev_percentage=0.5, show=False)
        azure.cut_skeleton_data(repetitions[0][0], repetitions[-1][1])
        azure.calculate_affine_transform_based_on_data(show=False)
        print(f"Agreement internal: {azure.check_agreement_of_both_cameras()}")

        azure.remove_unnecessary_joints()
        avg_df = azure.fuse_sub_and_master_cameras(alpha=0.1, window_size=7, show=True, path=None, joint="knee_left (y) ")

        viewer = SkeletonViewer()
        viewer.add_skeleton(azure.sub_position)
        viewer.add_skeleton(azure.mas_position)
        viewer.add_skeleton(avg_df)
        viewer.show_window()


if __name__ == '__main__':
    fuse_kinect_data(file_iterator.iterate_over_specific_subjects("CCB8AD"))
