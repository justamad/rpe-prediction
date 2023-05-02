from data_processing import iterate_segmented_data
from src.processing import segment_kinect_signal
from src.plot import evaluate_sample_predictions_individual
from os.path import exists

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt

src_path = "data/processed"
delta_t = 1.0 / 30  # Sampling Freq
m_wheel = 0.025  # kgm2
r_wheel = 0.135  # Radius of wheel in meters


def collect_data(file_path: str):
    train_set = pd.DataFrame()
    for trial in iterate_segmented_data(src_path, "full", plot=False, plot_path="data/plots"):
        rpe, subject, set_id, imu_df, pos_df, ori_df, hrv_df, flywheel_df = trial.values()
        con_reps, _ = segment_kinect_signal(pos_df["PELVIS (y)"], prominence=0.01, std_dev_p=0.4, min_dist_p=0.5,
                                            min_time=30, mode="concentric", show=False)
        ecc_reps, full_reps = segment_kinect_signal(pos_df["PELVIS (y)"], prominence=0.01, std_dev_p=0.4,
                                                    min_dist_p=0.5, min_time=30, mode="eccentric", show=False)

        durations_con, durations_ecc, durations_total = [], [], []
        mean_vel_con, mean_vel_ecc, mean_vel_total = [], [], []
        pos_df = pos_df / 1000  # Convert to meters
        for con, ecc, total in zip(con_reps, ecc_reps, full_reps):
            con_pos = pos_df["PELVIS (y)"][con[0]:con[1]].values
            ecc_pos = pos_df["PELVIS (y)"][ecc[0]:ecc[1]].values
            total_pos = pos_df["PELVIS (y)"][total[0]:total[1]].values

            durations_con.append(len(con_pos) * delta_t)
            durations_ecc.append(len(ecc_pos) * delta_t)
            durations_total.append(len(total_pos) * delta_t)

            mean_vel_con.append(np.abs(np.diff(con_pos) / delta_t).mean())
            mean_vel_ecc.append(np.abs(np.diff(ecc_pos) / delta_t).mean())
            mean_vel_total.append(np.abs(np.diff(total_pos) / delta_t).mean())

        flywheel_df = flywheel_df.copy()
        flywheel_df.loc[:, "durationCon"] = durations_con
        flywheel_df.loc[:, "durationEcc"] = durations_ecc
        flywheel_df.loc[:, "durationTotal"] = durations_total
        flywheel_df.loc[:, "velocityCon"] = mean_vel_con
        flywheel_df.loc[:, "velocityEcc"] = mean_vel_ecc
        flywheel_df.loc[:, "velocityTotal"] = mean_vel_total
        flywheel_df.loc[:, "subject"] = subject
        train_set = pd.concat([train_set, flywheel_df], axis=0, ignore_index=True)

    train_set.to_csv(file_path, index=False)


def calculate_power(mean_velocity: float, duration: float) -> float:
    power_kinect = m_wheel * mean_velocity ** 2 / (2 * r_wheel ** 2 * duration)
    return power_kinect


def calculate_correction(fw_power: np.ndarray, mean_vel: np.ndarray, durations: np.ndarray) -> float:
    predictions = []
    ground_truth = []
    # radii = []

    for idx in range(fw_power.shape[0]):
        ground_truth.append(fw_power)
        # radius = np.sqrt((m_wheel * mean_vel ** 2) / (2 * fw_power * fw_duration))
        # radii.append(radius)

        pred = calculate_power(mean_vel[idx], durations[idx])
        predictions.append(pred)

    mean_bias = np.mean(np.array(ground_truth) / np.array(predictions))
    # print(f"Radius Mean: {np.mean(radii)}, Radius Std: {np.std(radii)}")
    return float(mean_bias)


def train_physical_model(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    result_df = pd.DataFrame()
    subjects = df["subject"].unique()
    for subject in subjects:
        test_subject = df[df["subject"] == subject]
        train_subjects = df[df["subject"] != subject]
        bias = calculate_correction(
            train_subjects["power" + suffix].values,
            # train_subjects["duration"].values,
            train_subjects["velocity" + suffix].values,
            train_subjects["duration" + suffix].values,
        )
        predictions = [calculate_power(vel, dur) for vel, dur in zip(test_subject["velocity" + suffix], test_subject["duration" + suffix])]
        predictions = np.array(predictions) * bias

        # plt.plot(predictions, label="predictions")
        # plt.plot(test_subject["powerAvg"].values, label="ground_truth")
        # plt.legend()
        # plt.show()

        temp_df = {"prediction": predictions, "ground_truth": test_subject["powerAvg"]}
        temp_df = pd.DataFrame(temp_df)
        temp_df["subject"] = subject
        result_df = pd.concat([result_df, temp_df], axis=0, ignore_index=True)

    return result_df


if __name__ == "__main__":
    file_name = "data/training/physical_model.csv"
    if not exists(file_name):
        collect_data(file_name)
    df = pd.read_csv(file_name)
    exp = "Ecc"
    training_df = train_physical_model(df, exp)
    evaluate_sample_predictions_individual(training_df, "test", f"physics_results_{exp}")
