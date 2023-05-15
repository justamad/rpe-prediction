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
from src.dl import instantiate_best_dl_model, build_conv1d_model, build_cnn_lstm_model, WinDataGen
from src.dataset import dl_split_data
# from src.plot import (plot_sample_predictions)


def train_time_series_model(
        X: np.ndarray,
        y: pd.DataFrame,
        epochs: int,
        ground_truth: Union[List[str], str],
        win_size: int,
        batch_size: int,
):
    input_shape = (None, win_size, *X[0].shape[-2:])
    meta = {"X_shape_": input_shape, "n_outputs_": (None, 1)}
    model = build_cnn_lstm_model(meta=meta, kernel_size=11, n_filters=32, n_layers=3, dropout=0.5, lstm_units=32)
    model.summary()

    X_train, y_train, X_test, y_test = dl_split_data(X, y, ground_truth, 0.8)

    train_dataset = WinDataGen(X_train, y_train, win_size, 0.9, batch_size=batch_size, shuffle=True, balance=True)
    test_dataset = WinDataGen(X_test, y_test, win_size, 0.9, batch_size=1, shuffle=False, balance=False)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/fit/" + timestamp
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, callbacks=[tb_callback], verbose=1)
    model.save(f"models/{timestamp}/model")


def train_single_model(
        X: np.ndarray,
        y: pd.DataFrame,
        labels: str,
        epochs: int,
        batch_size: int,
):
    X_train, y_train, X_test, y_test = dl_split_data(X, y, label_col=labels, p_train=0.9)

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
            train_time_series_model(X, y, cfg["epochs"], cfg["labels"], win_size=30, batch_size=cfg["batch_size"])
        else:
            X = np.load(join(args.src_path, cfg["X_file"]))
            y = pd.read_csv(join(args.src_path, cfg["y_file"]))
            train_single_model(X, y, labels=cfg["labels"], epochs=cfg["epochs"], batch_size=cfg["batch_size"])
            # evaluate_single_model(X_train, y_train, X_test, y_test, src_path="models/20230511-100550/model")
