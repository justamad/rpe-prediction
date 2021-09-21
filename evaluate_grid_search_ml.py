from rpe_prediction.models import split_data_based_on_pseudonyms, evaluate_for_subject, normalize_rpe_values_min_max
from rpe_prediction.plot import plot_parallel_coordinates
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.svm import SVR
from scipy.stats import spearmanr, pearsonr
from os.path import join

import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str, dest='src_path', default="results/2021-09-17-14-24-36")
parser.add_argument('--data_path', type=str, dest='data_path', default="data/processed")
args = parser.parse_args()


def aggregate_features(input_path: str):
    files = filter(lambda x: x.startswith('features') and x.endswith('.csv'), os.listdir(input_path))
    data = []

    for file in files:
        df = pd.read_csv(join(input_path, file), delimiter=',', index_col=False).T
        df.rename(columns=df.iloc[0], inplace=True)
        df.drop(df.index[0], inplace=True)
        data.append(df)

    df = pd.concat(data)
    df_sum = df.sum(axis=0).sort_values(ascending=True) / len(data)
    df_sum.to_csv(join(input_path, "feature_ranks.csv"))


def evaluate_results_for_ml_model(input_path: str, ml_model: str = "svr"):
    results_data = []
    for file in filter(lambda x: x.startswith(ml_model) and x.endswith('.csv'), os.listdir(input_path)):
        split = file.split('_')
        win_size, overlap = int(split[2]), float(split[4][:-4])
        df = pd.read_csv(join(input_path, file), delimiter=';', index_col=False).sort_values(by='mean_test_R2',
                                                                                             ascending=True)

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

    plot_parallel_coordinates(
        results_data,
        color_column="mean_test_MAE",
        title=f"All parameters",
        file_name=f"total.png"
    )


def test_model(input_path: str, win_size: int, overlap: float):
    file = join(input_path, f"features_win_{win_size}_overlap_{overlap}.csv")
    features = pd.read_csv(file, delimiter=',', index_col=False)
    feature_mask = features[features['Rank'] == 1].loc[:, 'Unnamed: 0']

    X, y = pd.read_csv("data/X.csv", sep=';', index_col=False), pd.read_csv("data/y.csv", sep=';', index_col=False)
    y = normalize_rpe_values_min_max(y)
    X_train, y_train, X_test, y_test = split_data_based_on_pseudonyms(X, y, train_p=0.8, random_seed=19)
    X_train = X_train.loc[:, feature_mask]
    X_test = X_test.loc[:, feature_mask]

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='linear', gamma=0.001, C=1.0))
        # ('knn', KNeighborsRegressor(n_neighbors=5, weights='distance', leaf_size=10))
    ])
    pipe.fit(X_train, y_train['rpe'])

    train_pred = pipe.predict(X_train)
    test_pred = pipe.predict(X_test)
    y_test['prediction'] = test_pred

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

    subject_names = sorted(y_test['name'].unique())

    # Iterate over all test subjects
    for test_subject in subject_names:
        mask = y_test['name'] == test_subject
        s_test = y_test.loc[mask].copy()
        evaluate_for_subject(s_test)

    # plt.plot(test_pred)
    # plt.plot(y_test['rpe'].to_numpy())
    # plt.show()


if __name__ == '__main__':
    # aggregate_features(args.src_path)
    evaluate_results_for_ml_model(args.src_path, "svr")
    # test_model(args.src_path, 90, 0.7)
