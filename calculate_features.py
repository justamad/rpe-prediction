from argparse import ArgumentParser
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


class Generator(object):

    def __init__(self, path: str):
        self._path = path
        self._subjects = filter(lambda x: not x.startswith("."), os.listdir(path))

    def generate(self):
        for group_id, subject in enumerate(self._subjects):
            files = os.listdir(join(self._path, subject))
            files = sorted(filter(lambda x: "hrv" in x, files))
            for file in files:
                nr_set = int(file.split("_")[0])

                # azure = pd.read_csv(join(self._path, subject, file), sep=";", index_col="timestamp", parse_dates=True)
                hrv_df = pd.read_csv(join(self._path, subject, file), sep=";", index_col=0, parse_dates=True)
                # azure.index = pd.to_datetime(azure.index)

                dic = {
                    # "azure": azure,
                    "hrv": hrv_df,
                    "subject": subject,
                    "nr_set": nr_set,
                    "rpe": 15,
                    "group": group_id,
                }
                yield dic


def impute_dataframe(df: pd.DataFrame):
    df = df.interpolate(method="linear", limit_direction="forward", axis=0)
    df = df.fillna(0)
    return df


if __name__ == "__main__":
    window_size = 1
    overlap = 0.9

    gen = Generator("data/processed")
    a = gen.generate()

    X_data = pd.DataFrame()
    y_data = pd.DataFrame()

    for set_data in a:
        hrv = set_data["hrv"]
        hrv = impute_dataframe(hrv)
        X = calculate_statistical_features_with_sliding_window_time_based(
            [hrv],  # set_data["ecg"], set_data["imu"]],
            window_size=window_size,
            overlap=overlap,
        )

        y_values = [set_data["subject"], set_data["rpe"], set_data["group"], set_data["nr_set"]]
        y = pd.DataFrame(
            data=[y_values for _ in range(len(X))],
            columns=["name", "rpe", "group", "set"],
        )

        X_data = pd.concat([X_data, X], ignore_index=True)
        y_data = pd.concat([y_data, y], ignore_index=True)

    X_data.to_csv("X.csv", sep=";")
    y_data.to_csv("y.csv", sep=";")
