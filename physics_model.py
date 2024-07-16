import pandas as pd
import numpy as np
import matplotlib

from src.plot import (plot_sample_predictions, create_retrain_table, create_bland_altman_plot, create_scatter_plot)
from src.processing import segment_kinect_signal
from src.dataset import filter_labels_outliers_per_subject
from data_processing import iterate_segmented_data
from os.path import exists, join
from os import makedirs
from argparse import ArgumentParser
from typing import Tuple

DELTA_T = 1.0 / 30  # Sampling Freq
MASS_WHEEL = 0.025  # kgm2
RADIUS_WHEEL = 0.135  # Radius of wheel in meters
SUFFIX = "Avg"  # Use the average velocity and duration


def collect_data(file_path: str, src_path: str):
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

            durations_con.append(len(con_pos) * DELTA_T)
            durations_ecc.append(len(ecc_pos) * DELTA_T)
            durations_total.append(len(total_pos) * DELTA_T)

            mean_vel_con.append(np.abs(np.diff(con_pos) / DELTA_T).mean())
            mean_vel_ecc.append(np.abs(np.diff(ecc_pos) / DELTA_T).mean())
            mean_vel_total.append(np.abs(np.diff(total_pos) / DELTA_T).mean())

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


def calculate_power(mean_velocity: float, duration: float, correction_factor: float) -> float:
    return (MASS_WHEEL * mean_velocity ** 2) / (2 * RADIUS_WHEEL ** 2 * duration) * correction_factor ** 2


def calculate_radius(mean_velocity: float, duration: float, power: float):
    return np.sqrt((MASS_WHEEL * mean_velocity ** 2) / (2 * power * duration))


def calculate_correction_factor(fw_power: np.ndarray, mean_vel: np.ndarray, durations: np.ndarray) -> np.ndarray:
    corrections = []
    for gt_power, velocity, duration in zip(fw_power, mean_vel, durations):
        # kinect_power = calculate_power(velocity, duration, correction_factor=1.0)
        radius = calculate_radius(velocity, duration, gt_power)
        correction = RADIUS_WHEEL / radius  # Accordingly to formular
        # new_power = calculate_power(velocity, duration, correction_factor=correction)
        corrections.append(correction)

    return np.array(corrections)


def fit_model_cross_validation(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    result_df = pd.DataFrame()
    subject_biases = {}

    for idx, test_subject in enumerate(df["subject"].unique()):
        test_df = df[df["subject"] == test_subject]
        train_df = df[df["subject"] != test_subject]
        corrections = calculate_correction_factor(
            train_df["power" + SUFFIX].values,
            train_df["velocity" + SUFFIX].values,
            train_df["duration" + SUFFIX].values,
        )

        subject_biases[f"{idx}"] = (float(np.mean(corrections)), float(np.std(corrections)))
        predictions = [calculate_power(vel, dur, float(np.mean(corrections))) for vel, dur in
                       zip(test_df["velocity" + SUFFIX], test_df["duration" + SUFFIX])]

        temp_df = pd.DataFrame({"prediction": predictions, "ground_truth": test_df["powerAvg"]})
        temp_df["subject"] = test_subject
        result_df = pd.concat([result_df, temp_df], axis=0, ignore_index=True)

    biases_df = pd.DataFrame(subject_biases)
    return result_df, biases_df


def fit_model_globally(df: pd.DataFrame, bias: float) -> pd.DataFrame:
    predictions = [calculate_power(vel, dur, bias) for vel, dur in
                   zip(df["velocity" + SUFFIX], df["duration" + SUFFIX])]
    df["prediction"] = predictions
    df["ground_truth"] = df["powerAvg"]
    return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_file", type=str, dest="src_file", default="data/training/physical_model.csv")
    parser.add_argument("--src_path", type=str, dest="src_path", default="data/processed")
    parser.add_argument("--dst_path", type=str, dest="dst_path", default="data/physics_new")
    args = parser.parse_args()

    matplotlib.use("WebAgg")

    if not exists(args.src_file):
        collect_data(args.src_file, src_path=args.src_path)
    df = pd.read_csv(args.src_file)

    if not exists(args.dst_path):
        makedirs(args.dst_path)

    corrections = calculate_correction_factor(df["power" + SUFFIX].values, df["velocity" + SUFFIX].values, df["duration" + SUFFIX].values)
    print(f"Correction factor: {np.mean(corrections):.2f} +- {np.std(corrections):.2f}")

    df, _ = filter_labels_outliers_per_subject(df, df[["powerAvg", "subject"]], label_col="powerAvg")
    res_df, biases_df = fit_model_cross_validation(df)
    biases_df.index.name = "Subject"
    biases_df.T.to_latex(
        join(args.dst_path, "biases_latex.txt"),
        escape=False,
        float_format="%.2f",
        header=["Mean", "Std"],
        caption="Correction factors for each subject.",
    )

    # final_df.to_latex(join(dst_path, "retrain_results_latex.txt"), escape=False)
    biases_df.to_csv(join(args.dst_path, "biases.csv"), index=False)
    # res_df = calculate_power_globally(df, bias=float(np.mean(corrections)))

    res_df.to_csv(join(args.dst_path, "predictions.csv"), index=False)
    res_df["model"] = "Physics"
    physics_res_df = create_retrain_table(res_df, args.dst_path)
    physics_res_df.to_csv(join(args.dst_path, "results.csv"))

    create_bland_altman_plot(res_df, log_path=args.dst_path, file_name="physical_model")
    plot_sample_predictions(res_df, "poweravg", dst_path=args.dst_path)
    create_scatter_plot(res_df, log_path=args.dst_path, file_name="model", exp_name="poweravg")
