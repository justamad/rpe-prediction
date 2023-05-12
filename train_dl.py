import numpy as np
import pandas as pd
import logging
import tensorflow as tf
import yaml
import os
import matplotlib
import matplotlib.pyplot as plt

from typing import List, Union
from datetime import datetime
from argparse import ArgumentParser
from os.path import join, exists
from os import makedirs
from tensorflow import keras
from src.ml import MLOptimization
from src.dl import regression_models, instantiate_best_dl_model, build_conv1d_model, build_cnn_lstm_model
from src.dataset import dl_split_data
# from src.plot import (plot_sample_predictions)


def train_model(
        X: pd.DataFrame,
        y: pd.DataFrame,
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


def train_single_model(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int,
        batch_size: int,
):
    meta = {"X_shape_": X_train.shape, "n_outputs_": y_train.shape}
    # model = build_cnn_lstm_model(meta=meta, kernel_size=(11, 3), n_filters=32, n_layers=3, dropout=0.5, lstm_units=32)
    model = build_conv1d_model(meta=meta, kernel_size=3, n_filters=32, n_layers=3, dropout=0.5, n_units=128)
    model.summary()

    # Prepare the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_dataset = val_dataset.batch(batch_size)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/fit/" + timestamp
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[tb_callback])
    model.save(f"models/{timestamp}/model")


def evaluate_single_model(X_train, y_train, X_test, y_test, src_path: str):
    model = keras.models.load_model(src_path)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    fig, axs = plt.subplots(2, y_train.shape[1]) # , figsize=(15, 10))
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    # Plot Train
    for i in range(pred_train.shape[1]):
        axs[0, i].set_title("Train")
        axs[0, i].plot(pred_train[:, i], label="Prediction")
        axs[0, i].plot(y_train[:, i], label="Ground Truth")
        axs[0, i].legend()

    # Plot Test
    for i in range(pred_test.shape[1]):
        axs[1, i].set_title("Test")
        axs[1, i].plot(pred_test[:, i], label="Prediction")
        axs[1, i].plot(y_test[:, i], label="Ground Truth")
        axs[1, i].legend()

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
    parser.add_argument("--exp_file", type=str, dest="exp_file", default="experiments_dl/kinect.yaml")
    parser.add_argument("--train", type=bool, dest="train", default=True)
    parser.add_argument("--eval", type=bool, dest="eval", default=False)
    parser.add_argument("--single", type=bool, dest="single", default=True)
    parser.add_argument("--use_gpu", type=bool, dest="use_gpu", default=True)
    args = parser.parse_args()

    print(f"Available GPU devices: {tf.config.list_physical_devices('GPU')}")

    matplotlib.use("WebAgg")

    if args.train:
        if not args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        cfg = yaml.load(open(args.exp_file, "r"), Loader=yaml.FullLoader)

        X = np.load(join(args.src_path, cfg["X_file"]))
        y = pd.read_csv(join(args.src_path, cfg["y_file"]))

        if args.single:
            X_train, y_train, X_test, y_test = dl_split_data(X, y, label_col=cfg["labels"], p_train=0.9)
            train_single_model(X_train, y_train, X_test, y_test, epochs=cfg["epochs"], batch_size=cfg["batch_size"])
            # evaluate_single_model(X_train, y_train, X_test, y_test, src_path="models/20230511-100550/model")
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
