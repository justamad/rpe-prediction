import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
import os
import matplotlib

from typing import Dict
from src.dataset import dl_normalize_data_3d_subject, aggregate_results, normalize_labels_min_max
from src.dl import DLOptimization
from datetime import datetime
from argparse import ArgumentParser
from os.path import join, exists

from src.plot import (
    plot_sample_predictions,
    create_retrain_table,
    create_residual_plot,
    create_scatter_plot,
    create_bland_altman_plot,
)


def train_time_series_grid_search(X: np.ndarray, y: pd.DataFrame, dst_path: str, config: Dict[str, str]):
    with open(join(dst_path, "config.yml"), "w") as f:
        yaml.dump(config, f)

    opt = DLOptimization(X=X, y=y, **config)
    opt.perform_grid_search_with_cv(dst_path)


def evaluate_result_grid_search(src_path: str, aggregate: bool = False):
    dst_path = src_path.replace("train", "test")
    results_df = collect_trials(src_path)
    # results_df.rename(columns={"set_id": "ground_truth", "rpe": "set_id", "predictions": "prediction"}, inplace=True)
    results_df.rename(columns={"predictions": "prediction"}, inplace=True)
    plot_sample_predictions(results_df, "rpe", join(dst_path, "plots"))

    if aggregate:
        results_df["model"] = "CNN-GRU"
        results_df = aggregate_results(results_df)

    create_bland_altman_plot(results_df, join(dst_path), "CNN", "rpe")
    create_scatter_plot(results_df, dst_path, "CNN", "rpe")
    create_residual_plot(results_df, dst_path, "CNN")
    train_df = create_retrain_table(results_df, dst_path)
    train_df.to_csv(join(dst_path, "retrain.csv"))


def collect_trials(src_path: str) -> pd.DataFrame:
    result_df = pd.DataFrame()
    for folder in os.listdir(src_path):
        result_file = join(src_path, folder, "eval_dataset.csv")
        if not exists(result_file):
            continue

        df = pd.read_csv(result_file, index_col=0)
        result_df = pd.concat([result_df, df])

    return result_df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_path", type=str, dest="src_path", default="data/training")
    parser.add_argument("--eval_path", type=str, dest="eval_path", default="20231206-082601_kinect")
    parser.add_argument("--log_path", type=str, dest="log_path", default="results/dl/train")
    parser.add_argument("--exp_path", type=str, dest="exp_path", default="experiments/dl")
    parser.add_argument("--restore_path", type=str, dest="restore_path", default="20231206-141151_kinect")
    parser.add_argument("--exp_file", type=str, dest="exp_file", default="rpe_kinect.yaml")
    parser.add_argument("--train", type=bool, dest="train", default=True)
    parser.add_argument("--eval", type=bool, dest="eval", default=True)
    parser.add_argument("--use_gpu", type=bool, dest="use_gpu", default=True)
    args = parser.parse_args()

    matplotlib.use("WebAgg")
    print(f"Available GPU devices: {tf.config.list_physical_devices('GPU')}")

    if not args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    log_path = join(args.log_path, args.eval_path)

    if args.train:
        if args.restore_path:
            log_path = join(args.log_path, args.restore_path)
            cfg = yaml.load(open(join(log_path, "config.yml"), "r"), Loader=yaml.FullLoader)
            X = np.load(join(log_path, "X.npy"), allow_pickle=True)
            y = pd.read_csv(join(log_path, "y.csv"), index_col=0)
        else:
            cfg = yaml.load(open(join(args.exp_path, args.exp_file), "r"), Loader=yaml.FullLoader)
            target = args.exp_file.split("_")[1].replace(".yaml", "")
            log_path = join(args.log_path, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{target}")
            os.makedirs(log_path, exist_ok=True)

            X = np.load(join(args.src_path, cfg.pop("X_file")), allow_pickle=True)["X"]
            y = pd.read_csv(join(args.src_path, cfg.pop("y_file")), index_col=0)
            X = dl_normalize_data_3d_subject(X, y, method="std")

            np.save(join(log_path, "X.npy"), X)
            y.to_csv(join(log_path, "y.csv"))

        train_time_series_grid_search(X=X, y=y, dst_path=log_path, config=cfg)

    if args.eval:
        evaluate_result_grid_search(join(args.log_path, args.eval_path), aggregate=True)
