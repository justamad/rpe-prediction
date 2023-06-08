import pandas as pd
import numpy as np
import matplotlib

from data_processing import iterate_segmented_data
from src.processing import segment_kinect_signal
from src.dataset import filter_labels_outliers_per_subject
from os.path import exists, join
from os import makedirs
from argparse import ArgumentParser

from src.plot import (
    plot_sample_predictions,
    create_retrain_table,
    create_bland_altman_plot,
    create_scatter_plot,
)

delta_t = 1.0 / 30  # Sampling Freq
m_wheel = 0.025  # kgm2
r_wheel = 0.135  # Radius of wheel in meters


def collect_data(file_path: str, src_path: str = "data/processed"):
    train_set = pd.DataFrame()
    for trial in iterate_segmented_data(src_path, "full", plot=False, plot_path="data/plots"):
        meta_dict, imu_df, pos_df, ori_df, hrv_df, flywheel_df = trial.values()
        con_reps, _ = segment_kinect_signal(
            pos_df["PELVIS (y)"], prominence=0.01, std_dev_p=0.4, min_dist_p=0.5, min_time=30,
            mode="concentric", show=False,
        )
        ecc_reps, full_reps = segment_kinect_signal(
            pos_df["PELVIS (y)"], prominence=0.01, std_dev_p=0.4, min_dist_p=0.5, min_time=30,
            mode="eccentric", show=False,
        )

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
        for key, value in meta_dict.items():
            flywheel_df.loc[:, key] = value

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


def fit_predict_physical_model(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
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
    parser = ArgumentParser()
    parser.add_argument("--src_file", type=str, dest="src_file", default="data/training/physical_model.csv")
    parser.add_argument("--dst_path", type=str, dest="dst_path", default="data/physics")
    args = parser.parse_args()

    matplotlib.use("WebAgg")

    if not exists(args.src_file):
        collect_data(args.src_file)
    df = pd.read_csv(args.src_file)

    if not exists(args.dst_path):
        makedirs(args.dst_path)

    df, _ = filter_labels_outliers_per_subject(df, df[["powerAvg", "subject"]], label_col="powerAvg")
    res_df = fit_predict_physical_model(df, "Avg")
    res_df.to_csv(join(args.dst_path, "results.csv"), index=False)
    res_df["model"] = "Physics"
    physics_res_df = create_retrain_table(res_df, args.dst_path)
    physics_res_df.to_csv(join(args.dst_path, "results.csv"))

    create_bland_altman_plot(res_df, log_path=args.dst_path, file_name="Avg")
    plot_sample_predictions(res_df, "poweravg", dst_path=args.dst_path)
    create_scatter_plot(res_df, log_path=args.dst_path, file_name="model", exp_name="poweravg")
