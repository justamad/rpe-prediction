from src.processing import segment_signal_peak_detection
from src.dataset import SubjectDataIterator
from argparse import ArgumentParser
from os.path import join
from PyMoCapViewer import MoCapViewer

import numpy as np
import os.path
import pandas as pd
import logging
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt


def prepare_data_for_dl(dst_path: str):
    iterator = SubjectDataIterator(
        base_path=dst_path,
        log_path=args.dst_path,
        # dst_path=args.dst_path,
        data_loader=[
            SubjectDataIterator.STEREO_AZURE,
            SubjectDataIterator.RPE,
        ]
    )

    # total_df = pd.DataFrame()

    for set_id, trial in enumerate(iterator.iterate_over_all_subjects()):
        print("A new set")
        pos_df, rot_df = trial[SubjectDataIterator.STEREO_AZURE]
        pos_df.to_csv(join(trial["log_path"], "pos.csv"))
        rot_df.to_csv(join(trial["log_path"], "rot.csv"))

        # segments = segment_signal_peak_detection(pos_df["PELVIS (y)"], std_dev_p=0.2, show=False, log_path=None)
        # lengths = [s[1] - s[0] for s in segments]
        # cur_df = pd.DataFrame(lengths, columns=["length"])

        # viewer = MoCapViewer(sphere_radius=0.015, grid_axis=None)
        # viewer.add_skeleton(azure_df, skeleton_connection="azure")
        # viewer.show_window()

        # cur_df["rpe"] = trial[SubjectDataIterator.RPE]
        # cur_df["subject"] = trial["subject"]
        # cur_df["set_id"] = set_id
        # cur_df["nr_set"] = trial["nr_set"]
        # cur_df["group"] = trial["group"]

        # total_df = pd.concat([total_df, cur_df], ignore_index=True)

    # total_df.to_csv(join(dst_path, "durations.csv"))


# for trial in iterator.iterate_over_all_subjects():
#     azure = trial["azure"]
#     # azure.reduce_skeleton_joints()
#     azure_df = azure._fused
#
#     # Synchronize Faros <-> Kinect
#     physilog = trial["imu"]
#     # faros_imu = trial["ecg"]["imu"]
#     # hrv_df = trial["ecg"]["hrv"]
#
#     azure_acc_df = calculate_acceleration(azure_df)
#     shift_dt = calculate_cross_correlation_with_datetime(
#         reference_df=apply_butterworth_filter(physilog, cutoff=6, order=4, sampling_rate=128),
#         ref_sync_axis="CHEST_ACCELERATION_Z",
#         target_df=azure_acc_df,
#         target_sync_axis="SPINE_CHEST (y)",
#         show=args.show,
#         # log_path=join(trial["log_path"], "cross_correlation.png"),
#         log_path=join("plots/sync", f"{trial['subject_name']}_{trial['nr_set']}_cross_correlation.png"),
#     )
#     azure_acc_df.index += shift_dt
#     azure_df.index += shift_dt
#
#     output_path = join(args.log_path, trial["subject_name"], "azure")
#     create_folder_if_not_already_exists(output_path)
#     azure_df.to_csv(join(output_path, f"azure_{trial['nr_set'] + 1}.csv"))

    # repetitions = segment_1d_joint_on_example(
    #     joint_data=azure_df["PELVIS (y)"],
    #     exemplar=example,
    #     std_dev_p=0.5,
    #     show=False,
    #     log_path=join(trial["log_path"], "segmentation.png"),
    # )

    # Truncate data
    # cut_beginning = max(repetitions[0][0], physilog.index[0])
    # cut_end = min(repetitions[-1][1], physilog.index[-1])
    # azure_df = azure_df.loc[(azure_df.index > cut_beginning) & (azure_df.index < cut_end)]
    # physilog = physilog.loc[(physilog.index > cut_beginning) & (physilog.index < cut_end)]
    # ecg_df = ecg_df.loc[(ecg_df.index > cut_beginning) & (ecg_df.index < cut_end)]
    # faros_imu = faros_imu.loc[(faros_imu.index > cut_beginning) & (faros_imu.index < cut_end)]

    # fig, axs = plt.subplots(4, 1, sharex=True, figsize=(15, 12))
    # fig.suptitle(f"Subject: {trial['subject_name']}, Set: {trial['nr_set']}")
    # axs[0].plot(azure_df.filter(['SPINE_CHEST (x)', 'SPINE_CHEST (y)', 'SPINE_CHEST (z)'], axis=1))
    # axs[0].set_title("Kinect Position")
    # axs[1].plot(azure_acc_df.filter(['SPINE_CHEST (x)', 'SPINE_CHEST (y)', 'SPINE_CHEST (z)'], axis=1))
    # axs[1].set_title("Kinect Acceleration")
    # axs[2].plot(physilog.filter(['CHEST_ACCELERATION_X', 'CHEST_ACCELERATION_Y', 'CHEST_ACCELERATION_Z'], axis=1))
    # axs[2].set_title("Gaitup Acceleration")
    # # axs[2].plot(faros_imu, label=['X', 'Y', 'Z'])
    # # axs[2].set_title("Faros Acceleration")
    # # axs[3].plot(hrv_df)
    # # axs[3].set_title("Faros ECG")
    # # plt.savefig(join(trial['log_path'], "result.png"))
    # plt.show(block=True)
    # plt.close()
    # plt.cla()
    # plt.clf()

    # subject_path = join(args.dst_path, trial["subject_name"])
    # create_folder_if_not_already_exists(subject_path)
    #
    # azure_df.to_csv(join(subject_path, f"{trial['nr_set']:02d}_azure.csv"), sep=";")
    # hrv_df.to_csv(join(subject_path, f"{trial['nr_set']:02d}_hrv.csv"), sep=";")
    # physilog.to_csv(join(subject_path, f"{trial['nr_set']:02d}_imu.csv"), sep=";")
    # faros_imu.to_csv(join(subject_path, f"{trial['nr_set']:02d}_faros_imu.csv"), sep=";")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_path", type=str, dest="src_path", default="/media/ch/Data/RPE_Analysis")
    parser.add_argument("--log_path", type=str, dest="log_path", default="results")
    parser.add_argument("--dst_path", type=str, dest="dst_path", default="data/processed")
    parser.add_argument("--show", type=bool, dest="show", default=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)-8s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )

    if not os.path.exists(args.dst_path):
        os.makedirs(args.dst_path)

    prepare_data_for_dl(args.src_path)
