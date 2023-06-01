import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import yaml
import os
import matplotlib

from typing import List, Union
from datetime import datetime
from argparse import ArgumentParser
from os.path import join
from os import makedirs
from src.dl import build_cnn_lstm_model, WinDataGen, build_conv2d_model, ConvModelConfig, DLOptimization, CNNLSTMModelConfig, PerformancePlotCallback
from src.dataset import dl_split_data, filter_labels_outliers_per_subject, zero_pad_array, dl_normalize_data_3d_subject, dl_normalize_data_3d_global


def train_time_series_grid_search(X, y):
    opt = DLOptimization(X, y, balance=True, task="regression", mode="grid", ground_truth="rpe")
    opt.perform_grid_search_with_cv(CNNLSTMModelConfig(), "results_dl/rpe")


def train_time_series_model(
        X: np.ndarray,
        y: pd.DataFrame,
        epochs: int,
        ground_truth: Union[List[str], str],
        win_size: int,
        batch_size: int,
):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    makedirs(timestamp, exist_ok=True)

    meta = {"X_shape_": (None, win_size, 39, 3), "n_outputs_": (None, 1)}
    model = build_cnn_lstm_model(meta=meta, kernel_size=(3, 3), n_filters=64, n_layers=3, dropout=0.5, lstm_units=128)
    model.summary()

    X_train, y_train, X_test, y_test = dl_split_data(X, y, ground_truth, 0.7)

    train_dataset = WinDataGen(X_train, y_train, win_size, 0.9, batch_size=batch_size, shuffle=True, balance=True)
    test_dataset = WinDataGen(X_test, y_test, win_size, 0.5, batch_size=batch_size, shuffle=False, balance=False)
    val_dataset = WinDataGen(X_train, y_train, win_size, 0.5, batch_size=batch_size, shuffle=False, balance=False)

    performance_cb = PerformancePlotCallback(val_dataset, test_dataset, timestamp)

    # log_dir = "logs/fit/" + timestamp
    # tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, callbacks=[performance_cb])
    # model.save(f"models/{timestamp}/model")


def train_grid_search(X, y, labels):
    opt = DLOptimization(X, y, balance=False, task="regression", mode="grid", n_splits=16, ground_truth=labels)
    opt.perform_grid_search_with_cv(ConvModelConfig(), "results_dl/power")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_path", type=str, dest="src_path", default="data/training")
    parser.add_argument("--log_path", type=str, dest="log_path", default="results_dl")
    parser.add_argument("--exp_path", type=str, dest="exp_path", default="data/dl_experiments")
    parser.add_argument("--dst_path", type=str, dest="dst_path", default="evaluation_dl")
    parser.add_argument("--exp_file", type=str, dest="exp_file", default="kinect_lstm.yaml")
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

        cfg = yaml.load(open(join(args.exp_path, args.exp_file), "r"), Loader=yaml.FullLoader)

        if cfg["lstm"]:
            X = np.load(join(args.src_path, cfg["X_file"]), allow_pickle=True)["X"]
            y = pd.read_csv(join(args.src_path, cfg["y_file"]), index_col=0)
            X = dl_normalize_data_3d_subject(X, y, method="std")

            # for c in range(len(X)):
            #     d = X[c][:, :, 2]
            #     non_zero_columns = np.any(d != 0, axis=0)
            #     filtered_arr = d[:, non_zero_columns]
            #     X[c] = filtered_arr

            # X = dl_normalize_data_3d_subject(X, y, method="std")
            # X = dl_normalize_data_3d_global(X, method="min_max")

            train_time_series_model(X, y, cfg["epochs"], cfg["labels"], win_size=cfg["win_size"], batch_size=cfg["batch_size"])
            # train_time_series_grid_search(X, y)
        else:
            X = list(np.load(join(args.src_path, cfg["X_file"]), allow_pickle=True)["X"])  # TODO: check if list is necessary
            y = pd.read_csv(join(args.src_path, cfg["y_file"]))

            arr = np.vstack(X)
            mean = np.mean(arr, axis=0)
            std = np.std(arr, axis=0)

            for skeleton in range(len(X)):
                skel = (X[skeleton] - mean) / std
                X[skeleton] = zero_pad_array(skel, 170)

            X = np.array(X)
            X = np.nan_to_num(X)
            X, y = filter_labels_outliers_per_subject(X, y, cfg["labels"], sigma=3.0)

            train_grid_search(X, y, labels=cfg["labels"])
            # train_model_own_routine(X, y, labels=cfg["labels"], epochs=cfg["epochs"], batch_size=cfg["batch_size"], learning_rate=cfg["learning_rate"])
            # evaluate_single_model(X, y, src_path="models/20230519-115702/model")
