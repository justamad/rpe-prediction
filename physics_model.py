from data_processing import iterate_segmented_data
from src.processing import segment_kinect_signal
from src.plot import evaluate_sample_predictions_individual
from os.path import exists

import pandas as pd
import numpy as np

src_path = "data/processed"
dist_s = 1.0 / 30  # Sampling Freq
m_wheel = 0.025  # kgm2
radius = 0.135  # cm


def collect_data(file_path: str):
    train_set = pd.DataFrame()
    for trial in iterate_segmented_data(src_path, "full", plot=False, plot_path="data/plots"):
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

            durations_con.append(len(con_pos) * dist_s)
            durations_ecc.append(len(ecc_pos) * dist_s)
            durations_total.append(len(total_pos) * dist_s)

            mean_vel_con.append(np.abs(np.mean(np.diff(con_pos) * dist_s)))
            mean_vel_ecc.append(np.abs(np.mean(np.diff(ecc_pos) * dist_s)))
            mean_vel_total.append(np.abs(np.mean(np.diff(total_pos) * dist_s)))

        flywheel_df["duration_con"] = durations_con
        flywheel_df["duration_ecc"] = durations_ecc
        flywheel_df["duration_total"] = durations_total
        flywheel_df["mean_vel_con"] = mean_vel_con
        flywheel_df["mean_vel_ecc"] = mean_vel_ecc
        flywheel_df["mean_vel_total"] = mean_vel_total
        flywheel_df["subject"] = subject
        train_set = pd.concat([train_set, flywheel_df], axis=0, ignore_index=True)

    train_set.to_csv(file_path, index=False)


def apply_physics(mean_velo: np.ndarray, durations: np.ndarray, bias: float = 0) -> np.ndarray:
    predictions = []
    for idx in range(mean_velo.shape[0]):
        power = mean_velo[idx]
        duration = durations[idx]
        pred = 0 + bias
        predictions.append(pred)

    return np.ndarray(predictions)


def calculate_bias(fw_power, a_mean_vel, durations) -> float:
    predictions = []
    ground_truth = []
    differences = []
    for idx in range(fw_power.shape[0]):
        ground_truth.append(fw_power[idx])
        pred = apply_physics(fw_power[idx], a_mean_vel[idx], durations[idx])
        predictions.append(pred)
        differences.append(ground_truth / pred)

    mean_bias = np.mean(differences)
    return float(mean_bias)


def train_physical_model(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    result_df = pd.DataFrame()
    subjects = df["subject"].unique()
    for subject in subjects:
        test_subject = df[df["subject"] == subject]
        train_subjects = subjects[subjects != subject]

        c = calculate_bias(train_subjects["powerAvg"], train_subjects["mean_vel_total"], train_subjects["duration_total"])
        predictions = apply_physics(test_subject["mean_vel_total"], test_subject["durations_total"], c)
        temp_df = {
            "predictions": predictions,
            "ground_truth": test_subject["powerAvg"]
        }
        temp_df = pd.DataFrame(temp_df)
        result_df = pd.concat([result_df, temp_df], axis=0, ignore_index=True)

    return result_df


if __name__ == "__main__":
    file_name = "data/training/physical_model.csv"
    if not exists(file_name):
        collect_data(file_name)
    df = pd.read_csv(file_name, index_col=0)
    training_df = train_physical_model(df, "")
    evaluate_sample_predictions_individual(training_df, "test", ".")
