from src.camera import StereoAzure
from argparse import ArgumentParser
from os.path import join
from src.utils import create_folder_if_not_already_exists

from src.processing import (
    apply_butterworth_filter,
    calculate_acceleration,
    calculate_cross_correlation_with_datetime,
    segment_1d_joint_on_example,
)

from src.config import (
    SubjectDataIterator,
    ECGSubjectLoader,
    StereoAzureSubjectLoader,
    IMUSubjectLoader,
)

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("--src_path", type=str, dest="src_path", default="data/raw")
parser.add_argument("--log_path", type=str, dest="log_path", default="results")
parser.add_argument("--dst_path", type=str, dest="dst_path", default="data/processed")
parser.add_argument("--show", type=bool, dest="show", default=True)
args = parser.parse_args()

iterator = SubjectDataIterator(
    base_path=args.src_path,
    log_path=args.log_path,
    loaders=[StereoAzureSubjectLoader, ECGSubjectLoader, IMUSubjectLoader]
)

example = np.loadtxt("data/example.np")

for trial in iterator.iterate_over_all_subjects():
    print(trial)
    azure = StereoAzure(
        master_path=trial['azure']['master'],
        sub_path=trial['azure']['sub'],
    )
    azure.reduce_skeleton_joints()
    azure_df = azure.fuse_cameras(show=False)

    # Synchronize Faros <-> Kinect
    physilog = trial["imu"]
    faros_imu = trial["ecg"]["imu"]
    ecg_df = trial["ecg"]["ecg"]

    azure_acceleration = calculate_acceleration(azure_df)
    shift_dt = calculate_cross_correlation_with_datetime(
        # reference_df=apply_butterworth_filter(faros_imu, cutoff=4, order=4, sampling_rate=100),
        reference_df=apply_butterworth_filter(physilog, cutoff=4, order=4, sampling_rate=128),
        # reference_df=,
        # ref_sync_axis="ACCELERATION_X",
        ref_sync_axis="CHEST_ACCELERATION_Z",
        target_df=azure_acceleration * -1,
        target_sync_axis="SPINE_CHEST (y)",
        show=True,
        # log_path=join(trial['log_path'], "cross_correlation.png"),
    )
    azure_acceleration.index += shift_dt
    azure_df.index += shift_dt

    repetitions = segment_1d_joint_on_example(
        joint_data=azure_df["PELVIS (y)"],
        exemplar=example,
        std_dev_p=0.5,
        show=False,
        log_path=join(trial["log_path"], "segmentation.png"),
    )

    # Truncate data
    cut_beginning = max(repetitions[0][0], physilog.index[0])
    cut_end = min(repetitions[-1][1], physilog.index[-1])
    azure_df = azure_df.loc[(azure_df.index > cut_beginning) & (azure_df.index < cut_end)]
    physilog = physilog.loc[(physilog.index > cut_beginning) & (physilog.index < cut_end)]
    ecg_df = ecg_df.loc[(ecg_df.index > cut_beginning) & (ecg_df.index < cut_end)]
    faros_imu = faros_imu.loc[(faros_imu.index > cut_beginning) & (faros_imu.index < cut_end)]

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(15, 12))
    fig.suptitle(f"Subject: {trial['subject_name']}, Set: {trial['nr_set']}")
    axs[0].plot(azure_df.filter(['SPINE_CHEST (x)', 'SPINE_CHEST (y)', 'SPINE_CHEST (z)'], axis=1))
    axs[0].set_title('Kinect Acceleration')
    axs[1].plot(physilog.filter(['CHEST_ACCELERATION_X', 'CHEST_ACCELERATION_Y', 'CHEST_ACCELERATION_Z'], axis=1))
    axs[1].set_title('Gaitup Acceleration')
    axs[2].plot(faros_imu, label=['X', 'Y', 'Z'])
    axs[2].set_title('Faros Acceleration')
    axs[3].plot(ecg_df)
    axs[3].set_title('Neurokit2 HR')
    # plt.savefig(join(trial['log_path'], "result.png"))
    plt.show(block=True)
    plt.close()
    plt.cla()
    plt.clf()

    subject_path = join(args.dst_path, trial["subject_name"])
    create_folder_if_not_already_exists(subject_path)

    azure_df.to_csv(join(subject_path, f"{trial['nr_set']}_azure.csv"), sep=';')
    ecg_df.to_csv(join(subject_path, f"{trial['nr_set']}_ecg.csv"), sep=';')
    physilog.to_csv(join(subject_path, f"{trial['nr_set']}_physilog.csv"), sep=';')
