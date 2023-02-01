from src.dataset import SubjectDataIterator
from argparse import ArgumentParser
from typing import List
from os.path import join
from scipy import stats
from tsfresh.feature_extraction import MinimalFCParameters, extract_features, EfficientFCParameters, ComprehensiveFCParameters
from tsfresh.feature_extraction import ComprehensiveFCParameters, feature_calculators
from tsfresh.utilities.dataframe_functions import impute
# from PyMoCapViewer import MoCapViewer

from src.processing import (
    segment_signal_peak_detection,
    apply_butterworth_filter,
    calculate_acceleration,
    calculate_cross_correlation_with_datetime,
)

import numpy as np
import json
import os.path
import pandas as pd
import logging
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt


class CustomFeatures(ComprehensiveFCParameters):
    def __init__(self):
        ComprehensiveFCParameters.__init__(self)

        for f_name, f in feature_calculators.__dict__.items():
            is_minimal = (hasattr(f, "minimal") and getattr(f, "minimal"))
            is_curtosis_or_skew = f_name == "kurtosis" or f_name == "skewness"
            if f_name in self and not is_minimal and not is_curtosis_or_skew:
                del self[f_name]

        del self["sum_values"]
        del self["variance"]
        del self["mean"]
        # del self["length"]


# settings = MinimalFCParameters()
# settings = EfficientFCParameters()
# settings = ComprehensiveFCParameters()
# del settings["variance"]  # Variance and standard deviation are highly correlated but std integrates nr of samples
# del settings["sum_values"]  # Highly correlated with RMS and Mean
# del settings["mean"]  # Highly correlated with RMS and Sum
settings = CustomFeatures()


def truncate_data_frames(*data_frames) -> List[pd.DataFrame]:
    start_time = max([df.index[0] for df in data_frames])
    end_time = min([df.index[-1] for df in data_frames])

    result = [df[(df.index >= start_time) & (df.index < end_time)] for df in data_frames]
    # max_len = min([len(df) for df in result])
    # result = [df.iloc[:max_len] for df in result]
    return result


def mask_values_with_reps(df: pd.DataFrame, repetitions: List) -> pd.DataFrame:
    df["reps"] = -1
    for idx, (p1, p2) in enumerate(repetitions):
        df.loc[(df.index >= p1) & (df.index < p2), "reps"] = idx
    return df


def impute_dataframe(df: pd.DataFrame):
    df = df.interpolate(method="linear", limit_direction="forward", axis=0)
    df = df.fillna(0)
    return df


def filter_outliers_z_scores(df: pd.DataFrame, axis: str, z_score: float = 0.3) -> pd.DataFrame:
    z_scores = stats.zscore(df[axis])
    z_scores = z_scores.dropna(axis=1, how="all")
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < z_score).all(axis=1)
    return filtered_entries


def process_all_raw_data(src_path: str, dst_path: str, plot_path: str):
    iterator = SubjectDataIterator(
        base_path=src_path,
        dst_path=dst_path,
        data_loader=[
            SubjectDataIterator.FLYWHEEL,
            SubjectDataIterator.AZURE,
            SubjectDataIterator.IMU,
            SubjectDataIterator.HRV,
        ]
    )

    for set_id, trial in enumerate(iterator.iterate_over_all_subjects()):
        pos_df, ori_df = trial[SubjectDataIterator.AZURE]
        flywheel_df = trial[SubjectDataIterator.FLYWHEEL]
        hrv_df = trial[SubjectDataIterator.HRV]
        imu_df = trial[SubjectDataIterator.IMU]

        imu_df = apply_butterworth_filter(df=imu_df, cutoff=20, order=4, sampling_rate=128)
        azure_acc_df = calculate_acceleration(pos_df)
        shift_dt = calculate_cross_correlation_with_datetime(
            reference_df=imu_df,
            ref_sync_axis="CHEST_ACCELERATION_Z",
            target_df=azure_acc_df,
            target_sync_axis="SPINE_CHEST (y)",
            show=False,
        )
        azure_acc_df.index += shift_dt
        pos_df.index += shift_dt
        ori_df.index += shift_dt

        # imu_df, azure_acc_df, pos_df, hrv_df = truncate_data_frames(imu_df, azure_acc_df, pos_df, hrv_df)

        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(15, 12))
        fig.suptitle(f"Subject: {trial['subject']}, Set: {trial['nr_set']}")
        axs[0].plot(pos_df[['SPINE_CHEST (x)', 'SPINE_CHEST (y)', 'SPINE_CHEST (z)']])
        axs[0].set_title("Kinect Position")
        axs[1].plot(azure_acc_df[['SPINE_CHEST (x)', 'SPINE_CHEST (y)', 'SPINE_CHEST (z)']])
        axs[1].set_title("Kinect Acceleration")
        axs[2].plot(imu_df[['CHEST_ACCELERATION_X', 'CHEST_ACCELERATION_Y', 'CHEST_ACCELERATION_Z']])
        axs[2].set_title("Gaitup Acceleration")
        axs[3].plot(hrv_df[["Intensity (TRIMP/min)"]])
        axs[3].set_title("HRV")

        plt.savefig(join(plot_path, f"{trial['subject']}_{trial['nr_set']}.png"))
        # plt.show(block=True)
        plt.close()
        plt.cla()
        plt.clf()

        pos_df.to_csv(join(trial["dst_path"], "pos.csv"))
        ori_df.to_csv(join(trial["dst_path"], "ori.csv"))
        imu_df.to_csv(join(trial["dst_path"], "imu.csv"))
        hrv_df.to_csv(join(trial["dst_path"], "hrv.csv"))
        flywheel_df.to_csv(join(trial["dst_path"], "flywheel.csv"))


def prepare_segmented_data(src_path: str, dst_path: str, plot_path: str):
    final_df = pd.DataFrame()
    for subject in os.listdir(src_path):
        rpe_file = join(src_path, subject, "rpe_ratings.json")
        if not os.path.isfile(rpe_file):
            raise FileNotFoundError(f"Could not find RPE file for subject {subject}")

        with open(rpe_file) as f:
            rpe_values = json.load(f)
        rpe_values = {k: v for k, v in enumerate(rpe_values['rpe_ratings'])}

        subject_plot_path = join(plot_path, "segmented", subject)
        if not os.path.exists(subject_plot_path):
            os.makedirs(subject_plot_path)

        subject_path = join(src_path, subject)
        set_folders = list(filter(lambda x: x != "rpe_ratings.json", os.listdir(subject_path)))
        set_folders = sorted(map(lambda x: (int(x.split("_")[0]), join(subject_path, x)), set_folders))
        for set_id, set_folder in set_folders:
            logging.info(f"Processing subject {subject}, set {set_id}")

            # Read Dataframes
            imu_df = pd.read_csv(join(set_folder, "imu.csv"), index_col=0)
            imu_df.index = pd.to_datetime(imu_df.index)
            pos_df = pd.read_csv(join(set_folder, "pos.csv"), index_col=0)
            pos_df.index = pd.to_datetime(pos_df.index)
            ori_df = pd.read_csv(join(set_folder, "ori.csv"), index_col=0)
            ori_df.index = pd.to_datetime(ori_df.index)
            hrv_df = pd.read_csv(join(set_folder, "hrv.csv"), index_col=0)
            hrv_df.index = pd.to_datetime(hrv_df.index)
            flywheel_df = pd.read_csv(join(set_folder, "flywheel.csv"), index_col=0)
            flywheel_df = flywheel_df[flywheel_df["duration"] >= 1.5]

            # Segment signals
            # reps = segment_signal_peak_detection(pos_df["PELVIS (y)"], prominence=0.01, std_dev_p=0.2, show=False)
            imu_df_filter = apply_butterworth_filter(df=imu_df, cutoff=4, order=4, sampling_rate=128)
            reps = segment_signal_peak_detection(
                -imu_df_filter["CHEST_ACCELERATION_Z"],
                prominence=0.2,
                std_dev_p=0.7,
                show=False,
            )
            pos_df = mask_values_with_reps(pos_df, reps)
            ori_df = mask_values_with_reps(ori_df, reps)
            imu_df = mask_values_with_reps(imu_df, reps)
            hrv_df = mask_values_with_reps(hrv_df, reps)

            # viewer = MoCapViewer(sphere_radius=0.01)
            # viewer.add_skeleton(pos_df, skeleton_connection="azure")
            # viewer.show_window()

            fig, axs = plt.subplots(3, 1, sharex=True, figsize=(15, 12))
            axs[0].set_title(f"{len(flywheel_df)} vs. {len(reps)}")
            axs[0].plot(pos_df["PELVIS (y)"], color="gray")
            for p1, p2 in reps:
                axs[0].plot(pos_df["PELVIS (y)"][p1:p2])
                axs[0].plot(ori_df["KNEE_LEFT (x)"])

            axs[1].plot(imu_df["CHEST_ACCELERATION_Z"], color="gray")
            for p1, p2 in reps:
                axs[1].plot(imu_df["CHEST_ACCELERATION_Z"][p1:p2])

            axs[2].plot(hrv_df["Load (TRIMP)"], color="gray")
            for p1, p2 in reps:
                axs[2].plot(hrv_df["Load (TRIMP)"][p1:p2])

            # plt.show()
            plt.savefig(join(subject_plot_path, f"{subject}_{set_id}.png"))
            plt.clf()
            plt.cla()
            plt.close()

            pos_df = pos_df[pos_df["reps"] != -1]
            ori_df = ori_df[ori_df["reps"] != -1]
            imu_df = imu_df[imu_df["reps"] != -1]
            hrv_df = hrv_df[hrv_df["reps"] != -1]

            try:
                imu_features_df = extract_features(
                    timeseries_container=imu_df,
                    column_id="reps",
                    default_fc_parameters=settings,
                )
                imu_features_df = impute(imu_features_df)  # Replace Nan and inf by with extreme values (min, max)
                pos_features_df = extract_features(
                    timeseries_container=pos_df,
                    column_id="reps",
                    default_fc_parameters=settings,
                )
                pos_features_df = impute(pos_features_df)  # Replace Nan and inf by with extreme values (min, max)
                ori_features_df = extract_features(
                    timeseries_container=ori_df,
                    column_id="reps",
                    default_fc_parameters=settings,
                )
                ori_features_df = impute(ori_features_df)  # Replace Nan and inf by with extreme values (min, max)
                hrv_mean = hrv_df.groupby("reps").mean()

                # Match with FlyWheel data
                min_length = min(len(imu_features_df), len(pos_features_df), len(flywheel_df))
                total_df = pd.concat(
                    [
                        imu_features_df[:min_length].reset_index(drop=True).add_prefix("PHYSILOG_"),
                        pos_features_df[:min_length].reset_index(drop=True).add_prefix("KINECTPOS_"),
                        ori_features_df[:min_length].reset_index(drop=True).add_prefix("KINECTORI_"),
                        flywheel_df[:min_length].reset_index(drop=True).add_prefix("FLYWHEEL_"),
                        hrv_mean[:min_length].reset_index(drop=True).add_prefix("HRV_"),
                    ],
                    axis=1,
                )

                total_df["rpe"] = rpe_values[set_id]
                total_df["subject"] = subject
                total_df["set_id"] = set_id
                total_df["rep_id"] = total_df.index
                final_df = pd.concat([final_df, total_df], axis=0)

            except Exception as e:
                logging.error(f"Error while processing {subject} {set_id}: {e}")

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    final_df = impute_dataframe(final_df)
    final_df.reset_index(drop=True, inplace=True)
    final_df.to_csv(join(dst_path, "seg_hrv.csv"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_path", type=str, dest="src_path", default="/media/ch/Data/RPE_Analysis")
    parser.add_argument("--plot_path", type=str, dest="plot_path", default="plots")
    parser.add_argument("--dst_path", type=str, dest="dst_path", default="data")
    parser.add_argument("--show", type=bool, dest="show", default=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)-8s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )

    if not os.path.exists(args.dst_path):
        os.makedirs(args.dst_path)

    if not os.path.exists(args.plot_path):
        os.makedirs(args.plot_path)

    # process_all_raw_data(args.src_path, join(args.dst_path, "processed"), args.plot_path)
    prepare_segmented_data(join(args.dst_path, "processed"), join(args.dst_path, "training"), args.plot_path)
