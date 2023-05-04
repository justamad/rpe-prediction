from data_processing import iterate_segmented_data
from src.processing import segment_kinect_signal
from src.dataset import filter_labels_outliers_per_subject
from os.path import exists

from src.plot import (
    evaluate_sample_predictions_individual,
    create_retrain_table,
    create_bland_altman_plot,
    create_scatter_plot,
)

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
        flywheel_df.loc[:, "durationAvg"] = durations_total
        flywheel_df.loc[:, "velocityCon"] = mean_vel_con
        flywheel_df.loc[:, "velocityEcc"] = mean_vel_ecc
        flywheel_df.loc[:, "velocityAvg"] = mean_vel_total
        flywheel_df.loc[:, "subject"] = subject
        train_set = pd.concat([train_set, flywheel_df], axis=0, ignore_index=True)

    train_set.to_csv(file_path, index=False)


def calculate_power(mean_velocity: float, duration: float) -> float:
    delta = 0.3
    power_kinect = m_wheel * mean_velocity ** 2 / (2 * r_wheel ** 2 * (duration - delta))
    return power_kinect


def calculate_correction(fw_power: np.ndarray, mean_vel: np.ndarray, durations: np.ndarray) -> float:
    predictions = []
    ground_truth = []

    for idx in range(fw_power.shape[0]):
        ground_truth.append(fw_power)
        pred = calculate_power(mean_vel[idx], durations[idx])
        predictions.append(pred)

    mean_bias = np.mean(np.array(ground_truth) / np.array(predictions))
    return float(mean_bias)


def train_physical_model(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    result_df = pd.DataFrame()
    subjects = df["subject"].unique()
    for subject in subjects:
        test_subject = df[df["subject"] == subject]
        train_subjects = df[df["subject"] != subject]
        bias = calculate_correction(
            train_subjects["power" + suffix].values,
            train_subjects["velocity" + suffix].values,
            train_subjects["duration" + suffix].values,
        )
        predictions = [calculate_power(vel, dur) for vel, dur in zip(test_subject["velocity" + suffix], test_subject["duration" + suffix])]
        predictions = np.array(predictions) * bias

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

    train = True
    if train:
        for exp in ["Avg", "Con", "Ecc"]:
            y = df[["power" + exp, "subject"]]
            df, _ = filter_labels_outliers_per_subject(df, y, label_col="power" + exp)

            res_df = train_physical_model(df, exp)
            res_df.to_csv(f"results_{exp}.csv", index=False)
            res_df["model"] = "Physics"
            physics_res_df = create_retrain_table(res_df, ".")
            physics_res_df.to_csv(f"{exp}_results.csv")

            create_bland_altman_plot(res_df, log_path=".", file_name=exp)
            evaluate_sample_predictions_individual(res_df, "test", f"physics_results_{exp}")
            create_scatter_plot(res_df, log_path=".", file_name=exp)

    # Concentric
    con_df = pd.read_csv("evaluation/powercon/retrain_results.csv", index_col=0)
    physics_df = pd.read_csv("Con_results.csv", index_col=0)
    total_df = pd.concat([con_df, physics_df], axis=1, ignore_index=False)
    total_df.to_latex("Concentric_final.txt", escape=False, column_format="l" + "r" * (len(total_df.columns)))

    ecc_df = pd.read_csv("evaluation/powerecc/retrain_results.csv", index_col=0)
    physics_df = pd.read_csv("Ecc_results.csv", index_col=0)
    total_df = pd.concat([ecc_df, physics_df], axis=1, ignore_index=False)
    total_df.to_latex("Eccentric_final.txt", escape=False, column_format="l" + "r" * (len(total_df.columns)))

    avg_df = pd.read_csv("evaluation/poweravg/retrain_results.csv", index_col=0)
    physics_df = pd.read_csv("Avg_results.csv", index_col=0)
    total_df = pd.concat([avg_df, physics_df], axis=1, ignore_index=False)
    total_df.to_latex("Average_final.txt", escape=False, column_format="l" + "r" * (len(total_df.columns)))

