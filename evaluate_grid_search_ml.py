from rpe_prediction.plot import (
    plot_parallel_coordinates,
    plot_ml_predictions_for_sets,
    plot_ml_predictions_for_frames
)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from ast import literal_eval as make_tuple
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from scipy.stats import spearmanr, pearsonr
from os.path import join, isfile

import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str, dest='src_path', default="results/2021-10-15-10-46-02")
parser.add_argument('--model', type=str, dest='model', default="gbr")
args = parser.parse_args()


def create_folder_if_not_already_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def instantiate_mlp(p: pd.Series):
    return MLPRegressor(
        hidden_layer_sizes=make_tuple(p['param_mlp__hidden_layer_sizes']),
        activation=p['param_mlp__activation'],
        solver=p['param_mlp__solver'],
        learning_rate_init=p['param_mlp__learning_rate_init'],
        learning_rate=p['param_mlp__learning_rate'],
        max_iter=1000,
    )


def instantiate_gbr(p: pd.Series):
    return GradientBoostingRegressor(
        loss=p['param_gbr__loss'],
        learning_rate=p['param_gbr__learning_rate'],
        n_estimators=p['param_gbr__n_estimators'],
        n_iter_no_change=None if np.isnan(p['param_gbr__n_iter_no_change']) else int(p['param_gbr__n_iter_no_change'])
    )


models = {'svr': None,
          'gbr': instantiate_gbr,
          'mlp': instantiate_mlp}


def evaluate_best_performing_ml_model(input_path: str, ml_model: str = 'svr'):
    df = aggregate_individual_ml_trials_of_model(input_path, ml_model)
    best_combination = df.sort_values(by="mean_test_r2", ascending=False).iloc[0]
    win_size = best_combination[f'param_{ml_model}__win_size']
    overlap = best_combination[f'param_{ml_model}__overlap']

    output_path = join(input_path, ml_model, "results")
    create_folder_if_not_already_exists(output_path)

    X_train = pd.read_csv(join(input_path, f"X_train_win_{int(win_size)}_overlap_{overlap:.1f}.csv"), sep=';',
                          index_col=False)
    y_train = pd.read_csv(join(input_path, f"y_train_win_{int(win_size)}_overlap_{overlap:.1f}.csv"), sep=';',
                          index_col=False)
    X_test = pd.read_csv(join(input_path, f"X_test_win_{int(win_size)}_overlap_{overlap:.1f}.csv"), sep=';',
                         index_col=False)
    y_test = pd.read_csv(join(input_path, f"y_test_win_{int(win_size)}_overlap_{overlap:.1f}.csv"), sep=';',
                         index_col=False)

    model = models[ml_model](best_combination).fit(X_train, y_train['rpe'])

    augmented_mask = y_test['augmented'] == False
    X_test = X_test[augmented_mask]
    y_test = y_test[augmented_mask]

    augmented_mask = y_train['augmented'] == False
    X_train = X_train[augmented_mask]
    y_train = y_train[augmented_mask]

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    y_test['prediction'] = test_pred
    y_train['prediction'] = train_pred

    print(f"MSE Train: {mean_squared_error(y_train['rpe'], train_pred)}")
    print(f"MSE Test: {mean_squared_error(y_test['rpe'], test_pred)}")

    print(f"MAE Train: {mean_absolute_error(y_train['rpe'], train_pred)}")
    print(f"MAE Test: {mean_absolute_error(y_test['rpe'], test_pred)}")

    print(f"MAPE Train: {mean_absolute_percentage_error(y_train['rpe'], train_pred)}")
    print(f"MAPE Test: {mean_absolute_percentage_error(y_test['rpe'], test_pred)}")

    print(f"R2 Train: {r2_score(y_train['rpe'], train_pred)}")
    print(f"R2 Test: {r2_score(y_test['rpe'], test_pred)}")

    print(f"Spearman Correlation Train: {spearmanr(y_train['rpe'], train_pred)}")
    print(f"Spearman Correlation Test: {spearmanr(y_test['rpe'], test_pred)}")

    print(f"Pearson Correlation Train: {pearsonr(y_train['rpe'], train_pred)}")
    print(f"Pearson Correlation Test: {pearsonr(y_test['rpe'], test_pred)}")

    def plot_results(df, mode: str = "train"):
        for subject_name in sorted(df["name"].unique()):
            subject_df = df.loc[df["name"] == subject_name].copy()
            plot_ml_predictions_for_sets(
                subject_df,
                join(output_path, f"{mode}_subject_{subject_name}.png"),
            )

            plot_ml_predictions_for_frames(
                subject_df,
                join(output_path, f"pred_{mode}_subject_{subject_name}.png"),
            )

    plot_results(y_test, "test")
    plot_results(y_train, "train")


def aggregate_individual_ml_trials_of_model(input_path: str, ml_model: str = "svr"):
    file_name = join(input_path, ml_model, f"{ml_model}_results.csv")
    if isfile(file_name):
        df = pd.read_csv(file_name, sep=";", index_col=False)
        return df

    results_data = []

    for trial_file in filter(lambda x: x.endswith('csv'), os.listdir(join(input_path, ml_model))):
        split = trial_file.split('_')
        win_size, overlap = int(split[1]), float(split[3][:-4])
        df = pd.read_csv(join(input_path, ml_model, trial_file),
                         delimiter=';',
                         index_col=False).sort_values(by='mean_test_r2', ascending=True)

        plot_parallel_coordinates(
            df.copy(),
            color_column="mean_test_neg_mean_absolute_error",
            title=f"Window Size: {win_size}, Overlap: {overlap}",
            param_prefix=f"param_{ml_model}__",
            file_name=join(input_path, ml_model, f"window_size_{win_size}_overlap_{overlap}.png")
        )

        df.insert(0, f'param_{ml_model}__win_size', win_size)
        df.insert(1, f'param_{ml_model}__overlap', overlap)
        results_data.append(df)

    results_data = pd.concat(results_data, ignore_index=True).sort_values(by="mean_test_r2", ascending=True)
    results_data.to_csv(file_name, sep=';', index=False)

    plot_parallel_coordinates(
        results_data.copy(),
        color_column="mean_test_r2",
        title=f"All parameters",
        param_prefix=f"param_{ml_model}__",
        file_name=join(input_path, ml_model, f"total.png"),
    )

    return results_data


if __name__ == '__main__':
    evaluate_best_performing_ml_model(args.src_path, args.model)
