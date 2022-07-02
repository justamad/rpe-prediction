from datetime import datetime
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from os.path import join

from src.utils import (
    create_folder_if_not_already_exists,
    split_data_based_on_pseudonyms,
    normalize_rpe_values_min_max,
    filter_outliers_z_scores,
    normalize_data_by_subject,
)

from src.ml import (
    GridSearching,
    eliminate_features_with_xgboost_coefficients,
    eliminate_features_with_rfe,
    MLPModelConfig,
    GBRModelConfig,
    SVRModelConfig,
)

import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)-8s %(levelname)-8s %(message)s",
    datefmt="%m-%d %H:%M:%S",
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("my_logger").addHandler(console)

parser = ArgumentParser()
parser.add_argument("--src_path", type=str, dest="src_path", default="data/features")
parser.add_argument("--result_path", type=str, dest="result_path", default="results")
parser.add_argument("--nr_features", type=int, dest="nr_features", default=100)
parser.add_argument("--nr_augment", type=int, dest="nr_augment", default=0)
parser.add_argument("--borg_scale", type=int, dest="borg_scale", default=5)
args = parser.parse_args()


def impute_dataframe(df: pd.DataFrame):
    df = df.interpolate(method="linear", limit_direction="forward", axis=0)
    df = df.fillna(0)
    return df


def train_model(
        train_df: pd.DataFrame,
        log_path: str,
):
    X = train_df.iloc[:, :-4]
    y = train_df.iloc[:, -4:]

    X = eliminate_features_with_rfe(
        X_train=X,
        y_train=y["rpe"],
        step=10,
        nr_features=30,
    )

    # X["nr_set"] = y["nr_set"]

    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    for model_config in models:
        create_folder_if_not_already_exists(log_path)
        param_dict = model_config.get_trial_data_dict()
        grid_search = GridSearching(groups=y["group"], **param_dict)
        best_model, result_df = grid_search.perform_grid_search_with_cv(X, y["rpe"])
        result_df.to_csv(join(log_path, "result.csv"), sep=";")
        # logging.info(best_model.predict(X_test))
        # logging.info(best_model.score(X_test, y_test["rpe"]))


if __name__ == "__main__":
    models = [SVRModelConfig()]  # , GBRModelConfig()]

    for file in os.listdir(args.src_path):
        logging.info(f"Train on file: {file}")
        log_path = join(args.result_path, f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{file.replace('.csv', '')}")

        df = pd.read_csv(join(args.src_path, file), sep=";", index_col=0)

        mask = filter_outliers_z_scores(df)
        df = df[mask]
        df = impute_dataframe(df)
        df = normalize_data_by_subject(df)
        df = df.dropna(axis=1, how="all")
        df = impute_dataframe(df)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.mean(), inplace=True)
        # print(np.any(np.isnan(df)))
        # print(np.all(np.isfinite(df)))

        train_model(df, log_path)
