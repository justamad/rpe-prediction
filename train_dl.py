import numpy as np
import pandas as pd
import tensorflow as tf
import logging
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
from src.dl import instantiate_best_dl_model, build_conv1d_model, build_cnn_lstm_model, WinDataGen
from src.dataset import dl_split_data
# from src.plot import (plot_sample_predictions)


def train_time_series_model(
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        ground_truth: Union[List[str], str],
        win_size: int
        # search: str,
        # balancing: bool,
        # normalization_labels: str,
        # task: str,
        # n_splits: int,
):
    input_shape = (None, win_size, *X[0].shape[-2:])
    meta = {"X_shape_": input_shape, "n_outputs_": (None, 1)}
    model = build_cnn_lstm_model(meta=meta, kernel_size=(11, 3), n_filters=32, n_layers=3, dropout=0.5, lstm_units=32)
    model.summary()

    y = y[ground_truth].values
    train_dataset = WinDataGen(X, y, win_size, 0.5, 4, True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/fit/" + timestamp
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.fit(train_dataset, epochs=epochs, callbacks=[tb_callback])
    model.save(f"models/{timestamp}/model")

    # with open(join(log_path, "config.yml"), "w") as f:
    #     yaml.dump({
    #             "task": task,
    #             "search": search,
    #             "balancing": balancing,
    #             "normalization_labels": normalization_labels,
    #             "n_splits": n_splits,
    #         }, f)

    # MLOptimization(
    #     X=X,
    #     y=y,
    #     balance=False,
    #     task=task,
    #     mode=search,
    #     ground_truth=ground_truth,
    #     n_splits=n_splits,
    # ).perform_grid_search_with_cv(regression_models, log_path=log_path, n_jobs=1)


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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_path", type=str, dest="src_path", default="data/training")
    parser.add_argument("--log_path", type=str, dest="log_path", default="results_dl")
    parser.add_argument("--exp_path", type=str, dest="exp_path", default="experiments_dl")
    parser.add_argument("--dst_path", type=str, dest="dst_path", default="evaluation_dl")
    parser.add_argument("--exp_file", type=str, dest="exp_file", default="experiments_dl/kinect_lstm.yaml")
    parser.add_argument("--train", type=bool, dest="train", default=True)
    parser.add_argument("--eval", type=bool, dest="eval", default=False)
    parser.add_argument("--single", type=bool, dest="single", default=True)
    parser.add_argument("--use_gpu", type=bool, dest="use_gpu", default=True)
    args = parser.parse_args()
    matplotlib.use("WebAgg")

    print(f"Available GPU devices: {tf.config.list_physical_devices('GPU')}")

    if args.train:
        if not args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        cfg = yaml.load(open(args.exp_file, "r"), Loader=yaml.FullLoader)

        if cfg["lstm"]:
            X = np.load(join(args.src_path, cfg["X_file"]), allow_pickle=True)["X"]
            y = pd.read_csv(join(args.src_path, cfg["y_file"]), index_col=0)
            train_time_series_model(X, y, cfg["epochs"], cfg["labels"], win_size=30)
        else:
            X = np.load(join(args.src_path, cfg["X_file"]))
            y = pd.read_csv(join(args.src_path, cfg["y_file"]))

        if args.single:
            X_train, y_train, X_test, y_test = dl_split_data(X, y, label_col=cfg["labels"], p_train=0.9)
            train_single_model(X_train, y_train, X_test, y_test, epochs=cfg["epochs"], batch_size=cfg["batch_size"])
            # evaluate_single_model(X_train, y_train, X_test, y_test, src_path="models/20230511-100550/model")

        # else:
        #     for exp_name in os.listdir(args.exp_path):
        #         exp_path = join(args.exp_path, exp_name)
        #         for config_file in filter(lambda f: not f.startswith("_"), os.listdir(exp_path)):
        #             exp_config = yaml.load(open(join(exp_path, config_file), "r"), Loader=yaml.FullLoader)
        #             training_file = exp_config["training_file"]
        #             df = pd.read_csv(join(args.src_path, training_file), index_col=0, dtype={"subject": str})
        #             del exp_config["training_file"]
        #             seq_len = int(training_file.split("_")[0])
        #             log_path = join(args.log_path, f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
        #
        #             if not exists(log_path):
        #                 makedirs(log_path)
        #
        #             train_model(df, log_path, seq_len, **exp_config)

    # if args.eval:
    #     if not exists(args.dst_path):
    #         makedirs(args.dst_path)
    #
    #     evaluate_ml_model(join(args.log_path, "2023-04-20-16-33-17"), args.dst_path)
