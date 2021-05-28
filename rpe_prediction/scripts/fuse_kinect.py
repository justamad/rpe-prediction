from calibration import calculate_calibration
from rpe_prediction.config import SubjectDataIterator, KinectFusionLoaderSet
from rpe_prediction.processing import segment_1d_joint_on_example, apply_butterworth_df
from rpe_prediction.stereo_cam import StereoAzure
from rpe_prediction.rendering import SkeletonViewer
from os.path import join

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
        repetitions = segment_1d_joint_on_example(sub, example, min_duration=20, std_dev_percentage=0.5, show=True)
        azure.cut_skeleton_data(repetitions[0][0], repetitions[-1][1])
        azure.calculate_affine_transform_based_on_data(show=False)
        print(f"Agreement internal: {azure.check_agreement_of_both_cameras()}")

        azure.remove_unnecessary_joints()
        avg_df = azure.fuse_sub_and_master_cameras(alpha=0.1, window_size=5, show=True, path=None, joint="knee_left (y) ")
        avg_df_f = apply_butterworth_df(avg_df, order=4, sampling_frequency=30, fc=2)

        plt.plot(avg_df['pelvis (y) '])
        plt.plot(apply_butterworth_df(avg_df, order=4, sampling_frequency=30, fc=2)['pelvis (y) '], label="fc 2")
        plt.plot(apply_butterworth_df(avg_df, order=4, sampling_frequency=30, fc=4)['pelvis (y) '], label="fc 4")
        plt.plot(apply_butterworth_df(avg_df, order=4, sampling_frequency=30, fc=6)['pelvis (y) '], label="fc 6")
        plt.plot(apply_butterworth_df(avg_df, order=4, sampling_frequency=30, fc=8)['pelvis (y) '], label="fc 8")
        plt.legend()
        plt.show()

        viewer = SkeletonViewer()
        # viewer.add_skeleton(azure.sub_position)
        # viewer.add_skeleton(azure.mas_position)
        viewer.add_skeleton(avg_df_f)
        viewer.show_window()


if __name__ == '__main__':
    fuse_kinect_data(file_iterator.iterate_over_specific_subjects("CCB8AD"))
    # fuse_kinect_data(file_iterator.iterate_over_all_subjects())
