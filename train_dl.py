from src.ml import MLOptimization
from src.dl import regression_models, instantiate_best_dl_model
from typing import List, Union
from datetime import datetime
from argparse import ArgumentParser
from sklearn.metrics import r2_score
from os.path import join, exists
from os import makedirs
from tensorflow import keras

from src.plot import evaluate_aggregated_predictions, evaluate_sample_predictions, evaluate_sample_predictions_individual

from src.dataset import (
    extract_dataset_input_output,
    normalize_data_by_subject,
    normalize_data_global,
    filter_ground_truth_outliers,
)

import pandas as pd
import numpy as np
import logging
import tensorflow as tf
import yaml
import os
import time
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt


def train_model(
        df: pd.DataFrame,
        log_path: str,
        seq_length: int,
        drop_columns: List,
        drop_prefixes: List,
        ground_truth: Union[List[str], str],
        normalization_input: str,
        search: str,
        balancing: bool,
        normalization_labels: str,
        task: str,
        n_splits: int,
):
    if drop_prefixes is None:
        drop_prefixes = []
    if drop_columns is None:
        drop_columns = []

    X, y = extract_dataset_input_output(df=df, labels=ground_truth)
    for prefix in drop_prefixes:
        drop_columns += [col for col in df.columns if col.startswith(prefix)]

    X.drop(columns=drop_columns, inplace=True, errors="ignore")
    X = X.loc[:, (X != 0).any(axis=0)]  # Remove columns with all zeros, e.g. constrained joint orientations
    columns = X.columns

    if normalization_input == "subject":
        X = normalize_data_by_subject(X, y)
    elif normalization_input == "global":
        X = normalize_data_global(X)
    else:
        raise AttributeError(f"Unknown normalization method: {normalization_input}")

    X = X.values.reshape((-1, seq_length, X.shape[1]))
    y = y.iloc[::seq_length, :]
    X, y = filter_ground_truth_outliers(X, y, ground_truth)

    # Normalization labels
    label_mean, label_std = float('inf'), float('inf')
    if normalization_labels:
        values = y.loc[:, ground_truth].values
        label_mean, label_std = values.mean(axis=0), values.std(axis=0)
        y.loc[:, ground_truth] = (values - label_mean) / label_std

    pd.DataFrame(X.reshape(-1, X.shape[2]), columns=columns).to_csv(join(log_path, "X.csv"))
    y.to_csv(join(log_path, "y.csv"))

    with open(join(log_path, "config.yml"), "w") as f:
        yaml.dump(
            {
                "task": task,
                "search": search,
                "seq_length": seq_length,
                "drop_columns": drop_columns,
                "ground_truth": ground_truth,
                "drop_prefixes": drop_prefixes,
                "normalization_input": normalization_input,
                "balancing": balancing,
                "normalization_labels": normalization_labels,
                "label_mean": list(map(lambda v: float(v), label_mean)),
                "label_std": list(map(lambda v: float(v), label_std)),
                "n_splits": n_splits,
            },
            f,
        )

    MLOptimization(
        X=X,
        y=y,
        balance=False,
        task=task,
        mode=search,
        ground_truth=ground_truth,
        n_splits=n_splits,
    ).perform_grid_search_with_cv(regression_models, log_path=log_path, n_jobs=1)


def evaluate_ml_model(result_path: str, dst_path: str):
    config = yaml.load(open(join(result_path, "config.yml"), "r"), Loader=yaml.FullLoader)
    X = pd.read_csv(join(result_path, "X.csv"), index_col=0)
    y = pd.read_csv(join(result_path, "y.csv"), index_col=0)
    X = X.values.reshape((-1, config["seq_length"], X.shape[1]))

    for model_file in list(filter(lambda x: x.startswith("model__"), os.listdir(result_path))):
        model_name = model_file.replace("model__", "").replace(".csv", "")
        logging.info(f"Evaluating model: {model_name}")
        result_df = pd.read_csv(join(result_path, model_file))

        model = instantiate_best_dl_model(result_df, model_name=model_name, task=config["task"])

        opt = MLOptimization(
            X=X,
            y=y,
            balance=False,
            task=config["task"],
            mode=config["search"],
            ground_truth=config["ground_truth"],
            n_splits=config["n_splits"],
        )
        res_df = opt.evaluate_model(model, config["normalization_labels"], config["label_mean"], config["label_std"])
        res_df.to_csv(join(dst_path, model_name + ".csv"))

        # Evaluate multiple predictions
        for label_name in config["ground_truth"]:
            path = join(dst_path, model_name, label_name)
            if not exists(path):
                makedirs(path)

            name_dict = {
                f"{label_name}_ground_truth": "ground_truth",
                f"{label_name}_prediction": "prediction",
            }
            sub_df = res_df[list(name_dict.keys()) + ["subject"]].rename(columns=name_dict)
            evaluate_sample_predictions_individual(
                value_df=sub_df,
                gt_column="ground_truth",
                dst_path=path,
            )

        # evaluate_sample_predictions(
        #     result_dicts,
        #     config["ground_truth"],
        #     file_name=join(result_path, f"{model_name}_sample.png"),
        # )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_path", type=str, dest="src_path", default="data/training")
    parser.add_argument("--log_path", type=str, dest="log_path", default="results_dl")
    parser.add_argument("--exp_path", type=str, dest="exp_path", default="experiments_dl")
    parser.add_argument("--dst_path", type=str, dest="dst_path", default="evaluation_dl")
    parser.add_argument("--train", type=bool, dest="train", default=False)
    parser.add_argument("--eval", type=bool, dest="eval", default=True)
    parser.add_argument("--use_gpu", type=bool, dest="use_gpu", default=True)
    args = parser.parse_args()

    print(f"Available GPU devices: {tf.config.list_physical_devices('GPU')}")

    if args.train:
        if not args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        for exp_name in os.listdir(args.exp_path):
            exp_path = join(args.exp_path, exp_name)
            for config_file in filter(lambda f: not f.startswith("_"), os.listdir(exp_path)):
                exp_config = yaml.load(open(join(exp_path, config_file), "r"), Loader=yaml.FullLoader)
                training_file = exp_config["training_file"]
                df = pd.read_csv(join(args.src_path, training_file), index_col=0, dtype={"subject": str})
                del exp_config["training_file"]
                seq_len = int(training_file.split("_")[0])
                log_path = join(args.log_path, f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")

                if not exists(log_path):
                    makedirs(log_path)

                train_model(df, log_path, seq_len, **exp_config)

    if args.eval:
        if not exists(args.dst_path):
            makedirs(args.dst_path)

        evaluate_ml_model(join(args.log_path, "2023-04-18-10-50-26"), args.dst_path)
