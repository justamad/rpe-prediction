from src.config import ProcessedDataGenerator
from src.processing import segment_1d_joint_on_example
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

example = np.loadtxt("data/example.np")

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
    rep_counter = 0
    azure_data = pd.DataFrame()
    hrv_data = pd.DataFrame()
    imu_data = pd.DataFrame()
    y_data = pd.DataFrame()

    for set_data in data_gen.generate():
        print(set_data["nr_set"])
        azure_df = set_data["azure"]
        hrv_df = set_data["hrv"]
        imu_df = set_data["imu"]

        pelvis = azure_df["PELVIS (y)"]
        angles = calculate_angles_between_3_joints(azure_df)

        # plt.plot(pelvis, label="Azure")
        # plt.plot(imu_df["CHEST_ACCELERATION_Y"], label="IMU")
        # plt.plot(hrv_df["HRV_MeanNN"], label="HRV")
        # plt.show()

        segments = segment_1d_joint_on_example(pelvis, example, std_dev_p=1, show=False)
        first_rep = segments[0]
        template = pelvis[first_rep[0]: first_rep[1]]
        if set_data["subject"] not in subject_template:
            subject_template[set_data["subject"]] = template
        else:
            template = subject_template[set_data["subject"]]

        duration = []
        range = []
        dtw_cost = []

        for _, (start, end) in enumerate(segments):
            angles_sub = angles[start:end]
            angles_sub["id"] = rep_counter

            hrv_sub = hrv_df[start:end]
            hrv_sub["id"] = rep_counter

            # imu_sub = imu_df[start:end]
            # imu_sub["id"] = rep_counter

            rep_counter += 1

            # new = pelvis[start:end]
            # total_cost, warp_path = fastdtw(template, new)
            # dtw_cost.append(total_cost)
            # duration.append(len(new))
            # range.append(abs(max(new) - min(new)))
            azure_data = pd.concat([azure_data, angles_sub], ignore_index=True)
            hrv_data = pd.concat([hrv_data, hrv_sub], ignore_index=True)

        y = np.tile([set_data["nr_set"], set_data["subject"], set_data["rpe"], set_data["group"]], (len(segments), 1))
        y_df = pd.DataFrame(data=y, columns=["nr_set", "subject", "rpe", "group"])
        y_data = pd.concat([y_data, y_df], ignore_index=True)

    total_df = pd.DataFrame()
    for df in [azure_data, hrv_data]:
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
