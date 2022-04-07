from src.config import SubjectDataIterator, ECGSubjectLoader, RPESubjectLoader, AzureDataFrameLoader
from src.processing import calculate_acceleration, calculate_cross_correlation_with_datetime, apply_butterworth_filter

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


iterator = SubjectDataIterator(
    base_path="data/processed",
    log_path="data/results",
    loaders=[AzureDataFrameLoader, ECGSubjectLoader]
)

for set_data in iterator.iterate_over_all_subjects():
    azure_df = set_data["azure"]
    faros_imu = set_data["ecg"]["imu"]
    ecg = set_data["ecg"]["ecg"]

    azure_acceleration = calculate_acceleration(azure_df)
    shift_dt = calculate_cross_correlation_with_datetime(
        reference_df=apply_butterworth_filter(faros_imu, cutoff=4, order=4, sampling_rate=100),
        ref_sync_axis='ACCELERATION_X',
        target_df=azure_acceleration * -1,
        target_sync_axis='SPINE_CHEST (y)',
        show=True,
        # log_path=join(trial['log_path'], "cross_correlation.png"),
    )
    azure_acceleration.index += shift_dt
    azure_df.index += shift_dt

    print(len(faros_imu))
    print(len(ecg))
