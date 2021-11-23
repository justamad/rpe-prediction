from src.camera import StereoAzure
from src.utils import create_folder_if_not_already_exists
from argparse import ArgumentParser
from os.path import join

from src.processing import (
    resample_data,
    apply_butterworth_filter,
    calculate_acceleration,
    calculate_cross_correlation_with_datetime,
)

from src.config import (
    SubjectDataIterator,
    ECGSubjectLoader,
    StereoAzureSubjectLoader,
    RPESubjectLoader,
    IMUSubjectLoader,
)

import pandas as pd
# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument('--src_path', type=str, dest='src_path', default="data/raw")
parser.add_argument('--log_path', type=str, dest='log_path', default="data/processed")
args = parser.parse_args()

iterator = SubjectDataIterator(
    base_path=args.src_path,
    log_path=args.log_path,
    loaders=[StereoAzureSubjectLoader, ECGSubjectLoader, IMUSubjectLoader]
)

# for trial in iterator.iterate_over_specific_subjects("9AE368"):
for trial in iterator.iterate_over_all_subjects():
    azure = StereoAzure(
        master_path=trial['azure']['master'],
        sub_path=trial['azure']['sub'],
    )

    df = azure.fuse_cameras(show=False)
    df.index = pd.to_datetime(df.index, unit="s")

    subject_log_path = trial['log_path']
    create_folder_if_not_already_exists(subject_log_path)
    df.to_csv(join(subject_log_path, f"{trial['nr_set']}_azure.csv"), sep=';')

    # Synchronize Gaitup <-> Kinect
    gait_up = trial['imu']
    spine_chest = df.filter(['SPINE_CHEST (x)', 'SPINE_CHEST (y)', 'SPINE_CHEST (z)'], axis=1)
    spine_chest['SPINE_CHEST (y)'] *= -1
    spine_chest_acc = calculate_acceleration(spine_chest)
    shift_dt = calculate_cross_correlation_with_datetime(
        reference_df=gait_up,
        ref_sync_axis='CHEST_ACCELERATION_Y',
        target_df=spine_chest_acc,
        target_sync_axis='SPINE_CHEST (y)',
        show=False,
    )
    spine_chest_acc.index += shift_dt

    # Synchronize Gaitup <-> Faros
    imu = trial['ecg']['imu']
    shift_dt = calculate_cross_correlation_with_datetime(
        reference_df=gait_up,
        ref_sync_axis='CHEST_ACCELERATION_Y',
        target_df=imu,
        target_sync_axis='ACCELERATION_X',
        show=False,
    )

    # ECG data
    hr_df = trial['ecg']['hr']
    hr_df.index += shift_dt
    imu.index += shift_dt

    fig, axs = plt.subplots(4, 1, sharex=True)
    fig.suptitle(f"Subject: {trial['subject_name']}, Set: {trial['nr_set']}")
    axs[0].plot(spine_chest_acc)
    axs[0].set_title('Kinect Acceleration')
    axs[1].plot(gait_up["CHEST_ACCELERATION_Y"])
    axs[1].set_title('Gaitup Acceleration')
    axs[2].plot(imu, label=['X', 'Y', 'Z'])
    axs[2].set_title('Faros Acceleration')
    axs[3].plot(hr_df)
    axs[3].set_title('Neurokit2 HR')
    plt.show()
