from datetime import datetime
from os.path import join, isfile
from argparse import ArgumentParser

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
)

import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-8s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M:%S',
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


df = pd.read_csv("X.csv", sep=";", index_col=0)

mask = filter_outliers_z_scores(df)
df = impute_dataframe(df)
df = df[mask]
df = normalize_data_by_subject(df)
df = df.dropna(axis=1, how="all")


def train_model(train_df: pd.DataFrame):
    X, y = train_df.iloc[:, :-4], train_df.iloc[:, -4:]
    X = eliminate_features_with_rfe(
        X_train=X,
        y_train=y["rpe"],
        step=3,
        nr_features=5,
    )

    # X_train, y_train, X_test, y_test = split_data_based_on_pseudonyms(
    #     X=X,
    #     y=y,
    #     train_p=0.6,
    #     random_seed=42,
    # )

    for model_config in models:
        param_dict = model_config.get_trial_data_dict()
        grid_search = GridSearching(groups=y["group"], **param_dict)
        # file_name = join(result_path, str(model_config), f"win_{win_size}_overlap_{overlap}.csv")
        best_model = grid_search.perform_grid_search_with_cv(X, y["rpe"], output_file="output.csv")
        # logging.info(best_model.predict(X_test))
        # logging.info(best_model.score(X_test, y_test["rpe"]))


if __name__ == "__main__":
    models = [MLPModelConfig()]  # , GBRModelConfig()]

    result_path = join(args.result_path, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    # for model in models:
    #     create_folder_if_not_already_exists(join(result_path, str(model)))
    # create_folder_if_not_already_exists(args.feature_path)

    train_model(df)
