from src.dataset import ProcessedDataGenerator
from src.processing import segment_signal_peak_detection, apply_butterworth_filter
from src.features import calculate_angles_between_3_joints
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute
from fastdtw import fastdtw

import numpy as np
import pandas as pd
import tsfresh
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)-8s %(levelname)-8s %(message)s",
    datefmt="%m-%d %H:%M:%S",
)

settings = MinimalFCParameters()
del settings["variance"]  # Variance and standard deviation are highly correlated but std integrates nr of samples
# del settings['length']  # Length is constant for all windows
del settings["sum_values"]  # Highly correlated with RMS and Mean
# del settings['mean']  # Highly correlated with RMS and Sum


def impute_dataframe(df: pd.DataFrame):
    df = df.interpolate(method="linear", limit_direction="forward", axis=0)
    df = df.fillna(0)
    return df


if __name__ == '__main__':
    data_gen = ProcessedDataGenerator("data/processed")
    subject_template = {}
    features = pd.DataFrame()
    azure_data = pd.DataFrame()
    hrv_data = pd.DataFrame()
    imu_data = pd.DataFrame()
    y_data = pd.DataFrame()
    total_id = 0

    for set_data in data_gen.generate():
        logging.info(f"{set_data['subject']}: Set: {set_data['nr_set']}")
        azure_df = set_data["azure"]
        hrv_df = set_data["hrv"]
        imu_df = set_data["imu"]
        imu_df = apply_butterworth_filter(imu_df, sampling_rate=128, cutoff=5, order=4)

        # pelvis = azure_df["PELVIS (y)"]
        # angles = calculate_angles_between_3_joints(azure_df)

        chest = imu_df["CHEST_ACCELERATION_Z"]

        # plt.plot(chest, label="Azure")
        # plt.plot(imu_df["CHEST_ACCELERATION_Y"], label="IMU")
        # plt.plot(azure_df["PELVIS (y)"], label="Pelvis")
        # plt.plot(hrv_df["HRV_MeanNN"], label="HRV")
        # plt.legend()
        # plt.show()

        segments = segment_signal_peak_detection(chest, std_dev_p=0.6, show=False)
        if len(segments) == 0:
            continue

        first_rep = segments[0]
        template = chest[first_rep[0]: first_rep[1]]
        if set_data["subject"] not in subject_template:
            subject_template[set_data["subject"]] = template
        else:
            template = subject_template[set_data["subject"]]

        for rep_counter, (start, end) in enumerate(segments):
            # angles_sub = chest[start:end]
            # angles_sub["id"] = rep_counter

            hrv_sub = hrv_df.loc[start:end].copy()
            hrv_sub["id"] = total_id
            hrv_sub["rep"] = rep_counter

            imu_sub = imu_df[start:end].copy()
            imu_sub["id"] = total_id
            imu_sub["rep"] = rep_counter

            # azure_data = pd.concat([azure_data, angles_sub], ignore_index=True)
            hrv_data = pd.concat([hrv_data, hrv_sub], ignore_index=True)
            imu_data = pd.concat([imu_data, imu_sub], ignore_index=True)
            total_id += 1

        y_df = pd.DataFrame(
            data=np.tile(
                [set_data["nr_set"], set_data["subject"], set_data["rpe"], set_data["group"]],
                reps=(len(segments), 1)
            ),
            columns=["nr_set", "subject", "rpe", "group"]
        )
        y_data = pd.concat([y_data, y_df], ignore_index=True)

    total_df = pd.DataFrame()
    for df in [imu_data, hrv_data]:
        df = impute_dataframe(df)
        df = tsfresh.extract_features(
            timeseries_container=df,
            column_id="id",
            default_fc_parameters=settings,
        )
        df = impute(df)  # Replace Nan and inf by with extreme values (min, max)
        total_df = pd.concat([total_df, df], axis=1)

    total_df = pd.concat([total_df, y_data], axis=1)
    total_df.to_csv("bio_features.csv", index=False, sep=";")
