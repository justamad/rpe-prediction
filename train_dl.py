import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
import os
import matplotlib

from typing import List, Union
from datetime import datetime
from argparse import ArgumentParser
from os.path import join
from os import makedirs
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from scipy.stats import spearmanr
from src.dl import build_cnn_lstm_model, WinDataGen, ConvModelConfig, DLOptimization, CNNLSTMModelConfig, \
    PerformancePlotCallback
from src.dataset import dl_split_data, filter_labels_outliers_per_subject, zero_pad_array, dl_normalize_data_3d_subject, \
    dl_normalize_data_3d_global
from src.plot import plot_sample_predictions


def train_time_series_grid_search(X, y, label):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = join("data/dl_results", timestamp)
    opt = DLOptimization(X, y, balance=True, task="regression", mode="grid", ground_truth=label)
    opt.perform_grid_search_with_cv(CNNLSTMModelConfig(), log_path, lstm=True)


def evaluate_result_grid_search(src_path: str, dst_path: str):
    results_df = collect_trials(src_path)
    file = results_df.iloc[0]["file"]

    if not os.path.exists(dst_path):
        makedirs(dst_path)

    df = pd.read_csv(file, index_col=0)
    df.rename(columns={"y_true": "ground_truth", "y_pred": "prediction"}, inplace=True)  # TODO: fix this in the future
    plot_sample_predictions(df, "rpe", dst_path)


def collect_trials(dst_path: str) -> pd.DataFrame:
    data_dicts = []
    for folder in os.listdir(dst_path):
        result_file = join(dst_path, folder, "results.csv")
        if not os.path.exists(result_file):
            continue

        df = pd.read_csv(result_file, index_col=0)
        metrics = {
            "RMSE": lambda x, y: mean_squared_error(x, y, squared=False),
            "MAE": mean_absolute_error,
            "R2": r2_score,
            "MAPE": mean_absolute_percentage_error,
            "Spearman": lambda x, y: spearmanr(x, y)[0],
        }
        errors = {k: [] for k in metrics.keys()}
        for subject in df["subject"].unique():
            subject_df = df[df["subject"] == subject]

            for metric, func in metrics.items():
                errors[metric].append(func(subject_df["y_pred"], subject_df["y_true"]))

        errors = {k: np.mean(v) for k, v in errors.items()}
        errors["file"] = result_file

        params_dict = yaml.load(open(join(dst_path, folder, "params.yaml")), Loader=yaml.FullLoader)
        data_dicts.append({**params_dict, **errors})

    result_df = pd.DataFrame(data_dicts)
    result_df.sort_values(by="RMSE", inplace=True)
    return result_df


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
    model = build_cnn_lstm_model(kernel_size=(3, 3), n_filters=64, n_layers=3, dropout=0.5, lstm_units=128)
    model.summary()

    X_train, y_train, X_test, y_test = dl_split_data(X, y, ground_truth, 0.7)

    train_dataset = WinDataGen(X_train, y_train, win_size, 0.95, batch_size=batch_size, shuffle=True, balance=True)
    test_dataset = WinDataGen(X_test, y_test, win_size, 0.95, batch_size=batch_size, shuffle=False, balance=False)
    val_dataset = WinDataGen(X_train, y_train, win_size, 0.95, batch_size=batch_size, shuffle=False, balance=False)
    model.predict(val_dataset)
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
            X = dl_normalize_data_3d_subject(X, y, method="min_max")

            # for c in range(len(X)):
            #     d = X[c][:, :, 2]
            #     non_zero_columns = np.any(d != 0, axis=0)
            #     filtered_arr = d[:, non_zero_columns]
            #     X[c] = filtered_arr

            # X = dl_normalize_data_3d_subject(X, y, method="std")
            # X = dl_normalize_data_3d_global(X, method="min_max")

            # train_time_series_model(X, y, cfg["epochs"], cfg["labels"], 30, batch_size=cfg["batch_size"])
            train_time_series_grid_search(X, y, cfg["label"])
            # evaluate_result_grid_search("data/dl_results/20230605-211848/CNNLSTM", "data/dl_evaluation")
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
