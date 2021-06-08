from rpe_prediction.config import SubjectDataIterator, KinectFusionLoaderSet
from rpe_prediction.processing import segment_1d_joint_on_example
from rpe_prediction.stereo_cam import StereoAzure
from os.path import join

import numpy as np
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str, dest='src_path', default="data/raw")
parser.add_argument('--dst_path', type=str, dest='dst_path', default="data/processed/")
parser.add_argument('--show_plots', type=bool, dest='show_plots', default=False)
args = parser.parse_args()

example = np.loadtxt("example.np")


def fuse_kinect_data(iterator, show=False):
    """
    Fuse Kinect data from sub and master camera and calculate features
    @param iterator: iterator that crawls through raw data
    @param show: A flag whether to show the plots or not
    @return: None
    """
    for set_data in iterator:
        sub_path, master_path = set_data['azure']
        azure = StereoAzure(master_path=master_path, sub_path=sub_path)
        azure.reduce_skeleton_joints()
        print(f"Agreement initial: {azure.check_agreement_of_both_cameras()}")

        # Segment data based on pelvis joint
        sub = azure.sub_position['pelvis (y) '].to_numpy()
        repetitions = segment_1d_joint_on_example(sub, example, min_duration=20, std_dev_percentage=0.5, show=show)
        azure.cut_skeleton_data(repetitions[0][0], repetitions[-1][1])
        azure.calculate_affine_transform_based_on_data(show=show)
        print(f"Agreement internal: {azure.check_agreement_of_both_cameras()}")

        avg_df = azure.fuse_sub_and_master_cameras(alpha=0.1, window_size=5, show=show, path=None, joint="knee_left (y) ")

        subject = set_data['subject_name']
        current_set = set_data['nr_set']

        cur_path = join(args.dst_path, subject)
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)

        avg_df.to_csv(f"{os.path.join(cur_path, str(current_set))}_azure.csv", sep=';', index=True)

        # plt.plot(avg_df['pelvis (y) '])
        # plt.plot(apply_butterworth_df(avg_df, order=4, sampling_frequency=30, fc=2)['pelvis (y) '], label="fc 2")
        # plt.plot(apply_butterworth_df(avg_df, order=4, sampling_frequency=30, fc=4)['pelvis (y) '], label="fc 4")
        # plt.plot(apply_butterworth_df(avg_df, order=4, sampling_frequency=30, fc=6)['pelvis (y) '], label="fc 6")
        # plt.plot(apply_butterworth_df(avg_df, order=4, sampling_frequency=30, fc=8)['pelvis (y) '], label="fc 8")
        # plt.legend()
        # plt.show()

        # viewer = SkeletonViewer()
        # viewer.add_skeleton(azure.sub_position)
        # viewer.add_skeleton(azure.mas_position)
        # viewer.add_skeleton(avg_df)
        # viewer.show_window()


if __name__ == '__main__':
    if args.show_plots:  # Quick fix for running script on server
        import matplotlib
        matplotlib.use("TkAgg")

    file_iterator = SubjectDataIterator(args.src_path, KinectFusionLoaderSet())
    fuse_kinect_data(file_iterator.iterate_over_all_subjects(), show=args.show_plots)
