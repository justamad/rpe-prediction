from argparse import ArgumentParser
from src.config import ProcessedDataGenerator
from src.features import calculate_statistical_features_with_sliding_window_time_based
from os.path import join

import pandas as pd
import os

parser = ArgumentParser()
parser.add_argument("--src_path", type=str, dest="src_path", default="data/processed")
parser.add_argument("--dst_path", type=str, dest="dst_path", default="data/features")
parser.add_argument("--log_path", type=str, dest="log_path", default="results")
parser.add_argument("--show", type=bool, dest="show", default=True)
args = parser.parse_args()


def calculate_features(
        window_size: int,
        overlap: float
) -> pd.DataFrame:
    gen = ProcessedDataGenerator(args.src_path)
    df = pd.DataFrame()
    for set_data in gen.generate():
        print(set_data["subject"])
        print(set_data["nr_set"])
        hrv = set_data["hrv"]
        hrv = impute_dataframe(hrv)
        set_df = calculate_statistical_features_with_sliding_window_time_based(
            [hrv],  # set_data["ecg"], set_data["imu"]],
            window_size=window_size,
            overlap=overlap,
        )

        for column in ["subject", "rpe", "group", "nr_set"]:
           set_df[column] = set_data[column]

        # y_values = [set_data["subject"], set_data["rpe"], set_data["group"], set_data["nr_set"]]
        # y = pd.DataFrame(
        #     data=[y_values for _ in range(len(X))],
        #     columns=["name", "rpe", "group", "set"],
        # )

        df = pd.concat([df, set_df], ignore_index=True)

    return df


def impute_dataframe(df: pd.DataFrame):
    df = df.interpolate(method="linear", limit_direction="forward", axis=0)
    df = df.fillna(0)
    return df


if __name__ == "__main__":
    window_size = 1
    overlap = 0.9
    df = calculate_features(window_size=window_size, overlap=overlap)
    # plot_feature_distribution_as_pdf(
    # X_orig, X_scaled,
    #                                  join(result_path, f"features_win_{win_size}_overlap_{overlap}.pdf"))
    # X_scaled = filter_outliers_z_scores(X_orig)

    df.to_csv("X.csv", sep=";")
