from src.ml import MLOptimization
from src.dl import regression_models, instantiate_best_dl_model, build_conv_model
from src.dataset import normalize_labels_min_max, random_oversample
from typing import List, Union
from datetime import datetime
from argparse import ArgumentParser
from os.path import join, exists
from os import makedirs
from tensorflow import keras

from src.plot import (
    plot_sample_predictions,
)

from src.dataset import (
    extract_dataset_input_output,
    normalize_data_by_subject,
    normalize_data_global,
    filter_labels_outliers_per_subject,
)

import numpy as np
import pandas as pd
import logging
import tensorflow as tf
import yaml
import os
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
        X = normalize_data_by_subject(X, y, method="min_max")
    elif normalization_input == "global":
        X = normalize_data_global(X)
    else:
        raise AttributeError(f"Unknown normalization method: {normalization_input}")

    X = X.values.reshape((-1, seq_length, X.shape[1]))
    y = y.iloc[::seq_length, :]
    X, y = filter_labels_outliers_per_subject(X, y, ground_truth)

    # Normalization labels
    label_mean, label_std = float('inf'), float('inf')
    if normalization_labels:
        values = y.loc[:, ground_truth].values
        label_mean, label_std = values.mean(axis=0), values.std(axis=0)
        y.loc[:, ground_truth] = (values - label_mean) / label_std
        label_mean = list(map(lambda v: float(v), label_mean))
        label_std = list(map(lambda v: float(v), label_std))

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
                "label_mean": label_mean,
                "label_std": label_std,
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


def norm_data_3d(X: np.ndarray, y: pd.DataFrame, method="min_max"):
    for subject in y["subject"].unique():
        mask = y["subject"] == subject
        data = X[mask]
        if method == "min_max":
            min_, max_ = data.min(axis=0), data.max(axis=0)
            X[mask] = (data - min_) / (max_ - min_)
        else:
            mean, std = data.mean(axis=0), data.std(axis=0)
            X[mask] = (data - mean) / std

    return X


def process_data(X: pd.DataFrame, y: pd.DataFrame, label_col: str):
    X = X.loc[:, (X != 0).any(axis=0)]  # Remove columns with all zeros, e.g. constrained joint orientations
    X = X.values.reshape((-1, 45, X.shape[1]))
    # X, y = random_oversample(X, y, gt)
    # X = norm_data_3d(X, y, method="min_max")
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # y = normalize_labels_min_max(y, gt)
    return X, y


def split_data(X: np.ndarray, y: pd.DataFrame, label_col: str):
    subjects = y["subject"].unique()
    train_subjects = subjects[:int(len(subjects) * 0.8)]
    train_mask = y["subject"].isin(train_subjects)

    X_train, y_train = X[train_mask], y.loc[train_mask, :]
    X_test, y_test = X[~train_mask], y.loc[~train_mask, :]

    y_train = y_train[label_col].values
    y_test = y_test[label_col].values
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    return X_train, y_train, X_test, y_test


def train_single_model(X_train, y_train, X_test, y_test, epochs: int):
    meta = {"X_shape_": X_train.shape, "n_outputs_": 1}
    model = build_conv_model(meta=meta, kernel_size=(11, 3), n_filters=32, n_layers=3, dropout=0.5, n_units=128)
    model.summary()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/fit/" + timestamp
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_data=(X_test, y_test), callbacks=[tb_callback])
    model.save(f"models/{timestamp}/model")


def evaluate_single_model(X_train, y_train, X_test, y_test, src_path: str):
    model = keras.models.load_model(src_path)
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    axs[0].set_title("Train")
    axs[0].plot(model.predict(X_train), label="Prediction")
    axs[0].plot(y_train, label="Ground Truth")
    axs[0].legend()
    axs[1].set_title("Test")
    axs[1].plot(model.predict(X_test), label="Prediction")
    axs[1].plot(y_test, label="Ground Truth")
    axs[1].legend()
    plt.show()


def evaluate_ml_model(result_path: str, dst_path: str):
    dst_path = join(dst_path, os.path.basename(os.path.normpath(result_path)))

    if not exists(dst_path):
        makedirs(dst_path)

    config = yaml.load(open(join(result_path, "config.yml"), "r"), Loader=yaml.FullLoader)
    X = pd.read_csv(join(result_path, "X.csv"), index_col=0)
    y = pd.read_csv(join(result_path, "y.csv"), index_col=0)
    X = X.values.reshape((-1, config["seq_length"], X.shape[1]))

    for model_file in list(filter(lambda x: x.startswith("model__"), os.listdir(result_path))):
        model_name = model_file.replace("model__", "").replace(".csv", "")
        logging.info(f"Evaluating model: {model_name}")
        file_name = join(dst_path, f"{model_name}.csv")
        if exists(file_name):
            res_df = pd.read_csv(file_name, index_col=0)
        else:
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
            res_df = opt.evaluate_model(model, config["normalization_labels"], config["label_mean"],
                                        config["label_std"])
            res_df.to_csv(file_name)

        plot_sample_predictions(
            res_df, "hr", join(dst_path, model_name),
            pred_col="HRV_Mean HR (1/min)_prediction",
            label_col="HRV_Mean HR (1/min)_ground_truth",
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_path", type=str, dest="src_path", default="data/training")
    parser.add_argument("--log_path", type=str, dest="log_path", default="results_dl")
    parser.add_argument("--exp_path", type=str, dest="exp_path", default="experiments_dl")
    parser.add_argument("--dst_path", type=str, dest="dst_path", default="evaluation_dl")
    parser.add_argument("--train", type=bool, dest="train", default=True)
    parser.add_argument("--eval", type=bool, dest="eval", default=False)
    parser.add_argument("--single", type=bool, dest="single", default=True)
    parser.add_argument("--use_gpu", type=bool, dest="use_gpu", default=True)
    args = parser.parse_args()

    print(f"Available GPU devices: {tf.config.list_physical_devices('GPU')}")

    if args.train:
        if not args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        X = pd.read_parquet(join(args.src_path, "45_0.95_X.parquet"))
        X.drop(["Repetition"], axis=1, inplace=True)
        y = pd.read_parquet(join(args.src_path, "45_0.95_y.parquet"))
        # gt = "HRV_Mean HR (1/min)"
        labels = "FLYWHEEL_powerCon"
        X, y = process_data(X, y, label_col=labels)
        X_train, y_train, X_test, y_test = split_data(X, y, label_col=labels)

        if args.single:
            # train_single_model(X_train, y_train, X_test, y_test, epochs=100)
            evaluate_single_model(X_train, y_train, X_test, y_test, src_path="models/20230429-103728/model")
        else:
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

        evaluate_ml_model(join(args.log_path, "2023-04-20-16-33-17"), args.dst_path)
