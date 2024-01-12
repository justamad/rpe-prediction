import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
import os
import matplotlib

from typing import Dict
from src.dataset import dl_normalize_data_3d_subject, aggregate_results
from src.dl import DLOptimization
from datetime import datetime
from argparse import ArgumentParser
from os.path import join, exists, isfile

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
    final_results_df = pd.DataFrame()

    for model in results_df["model"].unique():
        model_df = results_df[results_df["model"] == model]
        cur_path = join(dst_path, model)
        os.makedirs(cur_path, exist_ok=True)

        plot_sample_predictions(model_df, "rpe", join(cur_path, "plots"))

        if aggregate:
            model_df = aggregate_results(model_df)

        create_bland_altman_plot(model_df, join(cur_path), "rpe")
        create_scatter_plot(model_df, cur_path, model, "rpe")
        create_residual_plot(model_df, cur_path, model)
        train_df = create_retrain_table(model_df, cur_path)
        final_results_df = pd.concat([final_results_df, train_df], axis=0)

    final_results_df.to_csv(join(dst_path, "retrain.csv"), index=False)


def collect_trials(src_path: str) -> pd.DataFrame:
    result_df = pd.DataFrame()
    for model_folder in os.listdir(src_path):
        if isfile(join(src_path, model_folder)):
            continue

        for folds in os.listdir(join(src_path, model_folder)):
            result_file = join(src_path, model_folder, folds, "eval_dataset.csv")
            if not exists(result_file):
                continue

            df = pd.read_csv(result_file, index_col=0)
            df["model"] = model_folder
            result_df = pd.concat([result_df, df])

    return result_df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_path", type=str, dest="src_path", default="data/training")
    parser.add_argument("--eval_path", type=str, dest="eval_path", default="Fusion")
    parser.add_argument("--dst_path", type=str, dest="dst_path", default="results/dl/train")
    parser.add_argument("--exp_path", type=str, dest="exp_path", default="experiments/dl")
    parser.add_argument("--restore_path", type=str, dest="restore_path", default=None)
    parser.add_argument("--exp_file", type=str, dest="exp_file", default="rpe_both.yaml")
    parser.add_argument("--train", type=bool, dest="train", default=False)
    parser.add_argument("--eval", type=bool, dest="eval", default=True)
    parser.add_argument("--use_gpu", type=bool, dest="use_gpu", default=True)
    args = parser.parse_args()

    matplotlib.use("WebAgg")
    print(f"Available GPU devices: {tf.config.list_physical_devices('GPU')}")

    if not args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    dst_path = join(args.dst_path, args.eval_path)

    if args.train:
        if args.restore_path and exists(join(args.dst_path, args.restore_path)):
            dst_path = join(args.dst_path, args.restore_path)
            cfg = yaml.load(open(join(dst_path, "config.yml"), "r"), Loader=yaml.FullLoader)
            X = np.load(join(dst_path, "X.npy"), allow_pickle=True)
            y = pd.read_csv(join(dst_path, "y.csv"), index_col=0)
        else:
            cfg = yaml.load(open(join(args.exp_path, args.exp_file), "r"), Loader=yaml.FullLoader)
            target = args.exp_file.split("_")[1].replace(".yaml", "")
            exp_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{target}"
            dst_path = join(args.dst_path, args.restore_path if args.restore_path else exp_name)
            os.makedirs(dst_path, exist_ok=True)

            X = np.load(join(args.src_path, cfg.pop("X_file")), allow_pickle=True)["X"]
            y = pd.read_csv(join(args.src_path, cfg.pop("y_file")), index_col=0)
            X = dl_normalize_data_3d_subject(X, y, method="std")

            np.save(join(dst_path, "X.npy"), X)
            y.to_csv(join(dst_path, "y.csv"))

        train_time_series_grid_search(X=X, y=y, dst_path=dst_path, config=cfg)

    if args.eval:
        for eva in ["Kinect", "IMU", "Fusion"]:
            evaluate_result_grid_search(join(args.dst_path, eva), aggregate=True)

        # evaluate_result_grid_search(join(args.dst_path, args.eval_path), aggregate=True)

        # def read_df(path):
        #     df = pd.read_csv(path, index_col=0)[["MSE_mean", "RMSE_mean", "MAPE_mean"]]
        #     df = df.rename(columns={"MSE_mean": "MSE", "RMSE_mean": "RMSE", "MAPE_mean": "MAPE"})
        #     return df
        #
        # fusion_df = read_df("results/dl/test/Fusion/retrain.csv")
        # imu_df = read_df("results/dl/test/IMU/retrain.csv")
        # kinect_df = read_df("results/dl/test/Kinect/retrain.csv")
        # print(fusion_df)
        # print(imu_df)
        # print(kinect_df)
        #
        # result_df = pd.concat([imu_df, kinect_df, fusion_df], axis=1, keys=["IMU", "Kinect", "Both"],
        #                       names=["Metrics", None])
        # result_df = result_df.swaplevel(0, 1, axis=1)
        # result_df = result_df.sort_index(axis=1, level=0, ascending=False)
        # result_df.to_latex("results/dl/test/final_results.txt", escape=False, float_format="%.2f")
        #
