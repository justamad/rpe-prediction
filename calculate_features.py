from src.config import IMUDataFrameLoader, AzureDataFrameLoader, ECGDataFrameLoader, SubjectDataIterator

from argparse import ArgumentParser

import logging

parser = ArgumentParser()
parser.add_argument("--src_path", type=str, dest="src_path", default="data/processed")
parser.add_argument("--dst_path", type=str, dest="dst_path", default="data/features")
parser.add_argument("--log_path", type=str, dest="log_path", default="results")
parser.add_argument("--show", type=bool, dest="show", default=True)
args = parser.parse_args()

iterator = SubjectDataIterator(
    base_path=args.src_path,
    log_path=args.log_path,
    loaders=[AzureDataFrameLoader, IMUDataFrameLoader, ECGDataFrameLoader]
)


def calculate_features():
    for trial in iterator.iterate_over_all_subjects():
        azure_df = trial["azure"]
        rr_df = trial["ecg"]
        imu_df = trial["imu"]

        # Truncate data
        # cut_beginning = max(repetitions[0][0], physilog.index[0])
        # cut_end = min(repetitions[-1][1], physilog.index[-1])
        # azure_df = azure_df.loc[(azure_df.index > cut_beginning) & (azure_df.index < cut_end)]
        # physilog = physilog.loc[(physilog.index > cut_beginning) & (physilog.index < cut_end)]
        # ecg_df = ecg_df.loc[(ecg_df.index > cut_beginning) & (ecg_df.index < cut_end)]
        # faros_imu = faros_imu.loc[(faros_imu.index > cut_beginning) & (faros_imu.index < cut_end)]


calculate_features()
