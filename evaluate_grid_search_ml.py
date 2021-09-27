from rpe_prediction.models import evaluate_for_subject
from rpe_prediction.plot import plot_parallel_coordinates
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from scipy.stats import spearmanr, pearsonr
from os.path import join

import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str, dest='src_path', default="results/2021-09-26-20-09-02")
args = parser.parse_args()


def instantiate_mlp(p: pd.Series):
    return MLPRegressor(
        # hidden_layer_sizes=p['param_mlp__hidden_layer_sizes'],
        hidden_layer_sizes=(100,),
        activation=p['param_mlp__activation'],
        solver=p['param_mlp__solver'],
        learning_rate_init=p['param_mlp__learning_rate_init'],
        learning_rate=p['param_mlp__learning_rate'],
        max_iter=1000,
    )


models = {'svr': None,
          'mlp': instantiate_mlp}


def evaluate_best_performing_ml_model(input_path: str, ml_model: str = 'svr'):
    df = aggregate_individual_ml_trials_of_model(input_path, ml_model, plot=False)
    best_combination = df.sort_values(by="mean_test_R2", ascending=False).iloc[0]
    win_size = best_combination['param__win_size']
    overlap = best_combination['param__overlap']

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

    for test_subject in sorted(y_test['name'].unique()):
        mask = y_test['name'] == test_subject
        s_test = y_test.loc[mask].copy()
        evaluate_for_subject(s_test)


def aggregate_individual_ml_trials_of_model(input_path: str, ml_model: str = "svr", plot: bool = True):
    results_data = []
    for file in filter(lambda x: x.startswith(ml_model) and x.endswith('.csv'), os.listdir(input_path)):
        split = file.split('_')
        win_size, overlap = int(split[2]), float(split[4][:-4])
        df = pd.read_csv(join(input_path, file), delimiter=';', index_col=False).sort_values(by='mean_test_R2',
                                                                                             ascending=True)

        if plot:
            mapping = {'linear': 0, 'rbf': 1}
            df = df[[c for c in df.columns if "param" in c or "mean_test" in c or "std_test" in c]]
            df = df.replace({'param_svr__kernel': mapping})
            plot_parallel_coordinates(
                df,
                color_column="mean_test_MAE",
                title=f"Window Size: {win_size}, Overlap: {overlap}",
                file_name=f"window_size_{win_size}_overlap_{overlap}.png"
            )

        df.insert(0, 'param__win_size', win_size)
        df.insert(1, 'param__overlap', overlap)
        results_data.append(df)

    results_data = pd.concat(results_data, ignore_index=True).sort_values(by="mean_test_R2", ascending=True)
    results_data.to_csv(f"{ml_model}_results.csv", sep=';', index=False)

    if plot:
        plot_parallel_coordinates(
            results_data,
            color_column="mean_test_MAE",
            title=f"All parameters",
            file_name=f"total.png"
        )

    return results_data


if __name__ == '__main__':
    evaluate_best_performing_ml_model(args.src_path, 'mlp')
