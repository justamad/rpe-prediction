from src.camera import StereoAzure
from src.utils import create_folder_if_not_already_exists
from argparse import ArgumentParser
from os.path import join

from src.processing import (
    resample_data,
    apply_butterworth_filter,
    calculate_acceleration,
    calculate_cross_correlation,
)

from src.config import (
    SubjectDataIterator,
    ECGSubjectLoader,
    StereoAzureSubjectLoader,
    RPESubjectLoader,
    IMUSubjectLoader,
)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument('--src_path', type=str, dest='src_path', default="data/raw")
parser.add_argument('--log_path', type=str, dest='log_path', default="data/processed")
args = parser.parse_args()

iterator = SubjectDataIterator(args.src_path, args.log_path).add_loader(IMUSubjectLoader)
for trial in iterator.iterate_over_all_subjects():
    print(trial)

# iterator = SubjectDataIterator(args.src_path, args.log_path).add_loader(StereoAzureSubjectLoader).add_loader(ECGSubjectLoader)
# # for trial in iterator.iterate_over_specific_subjects("9AE368"):
# for trial in iterator.iterate_over_all_subjects():
#     azure_paths = trial['azure']
#     azure = StereoAzure(master_path=azure_paths['master'], sub_path=azure_paths['sub'])
#     df = azure.fuse_cameras(show=False)
#     df.index = pd.to_datetime(df.index, unit="s")
#
#     subject_log_path = trial['log_path']
#     create_folder_if_not_already_exists(subject_log_path)
#     df.to_csv(join(subject_log_path, f"{trial['nr_set']}_azure.csv"), sep=';')
#
#     # IMU data
#     imu = trial['ecg'][1]
#     imu = imu.drop(columns=['Acceleration Magnitude'])
#     imu = apply_butterworth_filter(imu, cutoff=4, order=4, sampling_rate=100)
#
#     # Kinect data
#     spine_chest = df[['SPINE_CHEST (x)', 'SPINE_CHEST (y)', 'SPINE_CHEST (z)']]
#     spine_chest['SPINE_CHEST (y)'] *= -1
#     spine_chest = resample_data(spine_chest, 30, 100)
#     spine_chest.index = pd.to_datetime(spine_chest.index / 100, unit="s")
#     spine_chest_acc = calculate_acceleration(spine_chest)
#
#     spine_chest_sync = spine_chest_acc['SPINE_CHEST (y)']
#     imu_sync = imu['ACCELERATION_X'].to_numpy()
#
#     shift = calculate_cross_correlation(
#         reference_signal=spine_chest_sync,
#         target_signal=imu_sync,
#         sampling_frequency=100,
#     )
#
#     start_s = spine_chest.index[0]
#     start_d = imu.index[0]
#     shift_dt = (start_s - start_d) + pd.Timedelta(seconds=shift)
#
#     # ECG data
#     hr_df = trial['ecg'][2]
#     hr_df.index += shift_dt
#
#     # Correct clocks
#     imu.index += shift_dt
#
#     # fig, axs = plt.subplots(4, 1, sharex=True)
#     # fig.suptitle(trial['subject_name'])
#     # axs[0].plot(imu, label=['X', 'Y', 'Z'])
#     # axs[1].plot(hr_df)
#     # axs[2].plot(spine_chest_acc)
#     # plt.legend()
#     # plt.show()
