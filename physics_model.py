from data_processing import iterate_segmented_data
from src.processing import segment_kinect_signal
from os.path import exists

import pandas as pd
import numpy as np

src_path = "data/processed"
sampling_freq = 30


def collect_data(file_path: str):
    train_set = pd.DataFrame()
    for trial in iterate_segmented_data(src_path, "full", plot=False, plot_path="data/processed/segmentation"):
        rpe, subject, set_id, imu_df, pos_df, ori_df, hrv_df, flywheel_df = trial.values()
        con_reps, _ = segment_kinect_signal(pos_df["PELVIS (y)"], prominence=0.01, std_dev_p=0.4, min_dist_p=0.5,
                                            min_time=30, mode="concentric", show=False, )
        ecc_reps, full_reps = segment_kinect_signal(pos_df["PELVIS (y)"], prominence=0.01, std_dev_p=0.4,
                                                    min_dist_p=0.5, min_time=30, mode="eccentric", show=False, )

        durations_con, durations_ecc, durations_total = [], [], []
        mean_vel_con, mean_vel_ecc, mean_vel_total = [], [], []
        pos_df = pos_df / 1000  # Convert to meters
        for con, ecc, total in zip(con_reps, ecc_reps, full_reps):
            con_pos = pos_df["PELVIS (y)"][con[0]:con[1]].values
            ecc_pos = pos_df["PELVIS (y)"][ecc[0]:ecc[1]].values
            total_pos = pos_df["PELVIS (y)"][total[0]:total[1]].values

            durations_con.append(len(con_pos) / sampling_freq)
            durations_ecc.append(len(ecc_pos) / sampling_freq)
            durations_total.append(len(total_pos) / sampling_freq)

            mean_vel_con.append(np.abs(np.mean(np.diff(con_pos) / (1 / sampling_freq))))
            mean_vel_ecc.append(np.abs(np.mean(np.diff(ecc_pos) / (1 / sampling_freq))))
            mean_vel_total.append(np.abs(np.mean(np.diff(total_pos) / (1 / sampling_freq))))

        flywheel_df["duration_con"] = durations_con
        flywheel_df["duration_ecc"] = durations_ecc
        flywheel_df["duration_total"] = durations_total
        flywheel_df["mean_vel_con"] = mean_vel_con
        flywheel_df["mean_vel_ecc"] = mean_vel_ecc
        flywheel_df["mean_vel_total"] = mean_vel_total
        flywheel_df["subject"] = subject
        train_set = pd.concat([train_set, flywheel_df], axis=0, ignore_index=True)

    train_set.to_csv(file_path, index=False)


def apply_physics(df: pd.DataFrame) -> pd.DataFrame:
    pass


def calculate_bias(df: pd.DataFrame) -> float:
    pass


def train_physical_model(df: pd.DataFrame) -> pd.DataFrame:
    result_df = pd.DataFrame()
    subjects = df["subject"].unique()
    for subject in subjects:
        test_subject = df[df["subject"] == subject]
        train_subjects = subjects[subjects != subject]

        constant_value = calculate_bias(train_subjects)
        apply_physics(test_subject)

    return result_df


if __name__ == "__main__":
    file_name = "data/processed/kinect_flywheel.csv"
    if not exists(file_name):
        collect_data(file_name)
    df = pd.read_csv(file_name)

