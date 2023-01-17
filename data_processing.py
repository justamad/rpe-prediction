from src.dataset import SubjectDataIterator
from argparse import ArgumentParser
from typing import List
from os.path import join
from PyMoCapViewer import MoCapViewer

from src.processing import (
    segment_signal_peak_detection,
    apply_butterworth_filter,
    calculate_acceleration,
    calculate_cross_correlation_with_datetime,
)

import numpy as np
import json
import os.path
import pandas as pd
import logging
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt


def truncate_data_frames(*data_frames) -> List[pd.DataFrame]:
    start_time = max([df.index[0] for df in data_frames])
    end_time = min([df.index[-1] for df in data_frames])

    result = [df[(df.index >= start_time) & (df.index < end_time)] for df in data_frames]
    # max_len = min([len(df) for df in result])
    # result = [df.iloc[:max_len] for df in result]
    return result


def process_all_raw_data(src_path: str, dst_path: str, plot_path: str):
    iterator = SubjectDataIterator(
        base_path=src_path,
        dst_path=dst_path,
        data_loader=[
            SubjectDataIterator.AZURE,
            SubjectDataIterator.IMU,
            SubjectDataIterator.HRV,
        ]
    )

    for set_id, trial in enumerate(iterator.iterate_over_all_subjects()):
        pos_df = trial[SubjectDataIterator.AZURE]

        imu_df = trial[SubjectDataIterator.IMU]
        imu_df = apply_butterworth_filter(df=imu_df, cutoff=20, order=4, sampling_rate=128)

        azure_acc_df = calculate_acceleration(pos_df)
        shift_dt = calculate_cross_correlation_with_datetime(
            reference_df=imu_df,
            ref_sync_axis="CHEST_ACCELERATION_Z",
            target_df=azure_acc_df,
            target_sync_axis="SPINE_CHEST (y)",
            show=False,
        )
        azure_acc_df.index += shift_dt
        pos_df.index += shift_dt

        hrv_df = trial[SubjectDataIterator.HRV]
        imu_filter_df, azure_acc_df, pos_df, hrv_df = truncate_data_frames(imu_df, azure_acc_df, pos_df, hrv_df)

        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(15, 12))
        fig.suptitle(f"Subject: {trial['subject']}, Set: {trial['nr_set']}")
        axs[0].plot(pos_df[['SPINE_CHEST (x)', 'SPINE_CHEST (y)', 'SPINE_CHEST (z)']])
        axs[0].set_title("Kinect Position")
        axs[1].plot(azure_acc_df[['SPINE_CHEST (x)', 'SPINE_CHEST (y)', 'SPINE_CHEST (z)']])
        axs[1].set_title("Kinect Acceleration")
        axs[2].plot(imu_filter_df[['CHEST_ACCELERATION_X', 'CHEST_ACCELERATION_Y', 'CHEST_ACCELERATION_Z']])
        axs[2].set_title("Gaitup Acceleration")
        axs[3].plot(hrv_df[["Intensity (TRIMP/min)"]])
        axs[3].set_title("HRV")

        plt.savefig(join(plot_path, f"{trial['subject']}_{trial['nr_set']}.png"))
        # plt.show(block=True)
        plt.close()
        plt.cla()
        plt.clf()

        pos_df.to_csv(join(trial["dst_path"], "pos.csv"))
        imu_df.to_csv(join(trial["dst_path"], "imu.csv"))
        hrv_df.to_csv(join(trial["dst_path"], "hrv.csv"))
        # rot_df.to_csv(join(trial["log_path"], "rot.csv"))


def prepare_data_for_deep_learning(src_path: str, dst_path: str):
    total_df = pd.DataFrame()
    set_counter = 0

    for subject in os.listdir(src_path):
        subject_path = join(src_path, subject)

        if os.path.isfile(subject_path):
            continue

        with open(join(subject_path, "rpe_ratings.json")) as f:
            rpe_values = json.load(f)
        rpe_values = {k: v for k, v in enumerate(rpe_values['rpe_ratings'])}

        for set_id in os.listdir(subject_path):
            if os.path.isfile(join(subject_path, set_id)):
                continue

            logging.info(f"Processing subject {subject}, set {set_id}")
            n_set = int(set_id.split("_")[0])
            set_path = join(subject_path, set_id)
            imu_df = pd.read_csv(join(set_path, "imu.csv"), index_col=0)
            # ori_df = pd.read_csv(join(set_path, "rot.csv"), sep=",", index_col=0)
            # ori_df["rpe"] = rpe_values[n_set]
            # ori_df["subject"] = subject
            # ori_df["set_id"] = set_counter
            imu_df["rpe"] = rpe_values[n_set]
            imu_df["subject"] = subject
            imu_df["set_id"] = set_counter

            total_df = pd.concat([total_df, imu_df], axis=0)
            set_counter += 1

    total_df.to_csv(join(dst_path, "dl_imu.csv"))


def prepare_conventional_machine_learning(src_path: str, dst_path: str):
    total_df = pd.DataFrame()
    set_counter = 0

    for subject in os.listdir(src_path):
        subject_path = join(src_path, subject)

        if os.path.isfile(subject):
            continue

        # repetitions = segment_1d_joint_on_example(
        #     joint_data=azure_df["PELVIS (y)"],
        #     exemplar=example,
        #     std_dev_p=0.5,
        #     show=False,
        #     log_path=join(trial["log_path"], "segmentation.png"),
        # )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_path", type=str, dest="src_path", default="/media/ch/Data/RPE_Analysis")
    parser.add_argument("--plot_path", type=str, dest="plot_path", default="plots")
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

    if not os.path.exists(args.plot_path):
        os.makedirs(args.plot_path)

    process_all_raw_data(args.src_path, args.dst_path, args.plot_path)
    # prepare_data_for_deep_learning(args.dst_path, args.dst_path)
