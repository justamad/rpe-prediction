from rpe_prediction.models import split_data_to_pseudonyms, evaluate_for_subject
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.svm import SVR
from scipy.stats import spearmanr, pearsonr
from sklearn.neighbors import KNeighborsRegressor
from os.path import join

import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str, dest='src_path', default="results/2021-06-14-13-15-57")
parser.add_argument('--data_path', type=str, dest='data_path', default="data/processed")
args = parser.parse_args()


def aggregate_features(input_path):
    """
    Aggregates the features from all trials into one single csv file
    @param input_path: The current input file where files reside in
    :return: None
    """
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


def analyze_svr_results(input_path):
    """
    Read in results from classifiers and concatenate into large data frame
    :param input_path: the path to the result data
    :return: None
    """
    data = []
    for file in filter(lambda x: x.startswith('svr') and x.endswith('.csv'), os.listdir(input_path)):
        split = file.split('_')
        classifier, win_size, overlap = split[0], int(split[2]), float(split[4][:-4])
        df = pd.read_csv(join(input_path, file), delimiter=';', index_col=False)
        df.insert(0, 'model', classifier)
        df.insert(1, 'win_size', win_size)
        df.insert(2, 'overlap', overlap)
        data.append(df)

    data = pd.concat(data)
    data.to_csv(join(input_path, "classifier_results.csv"), sep=';', index=False)


def test_model(input_path, win_size, overlap):
    file = join(input_path, f"features_win_{win_size}_overlap_{overlap}.csv")
    features = pd.read_csv(file, delimiter=',', index_col=False)
    feature_mask = features[features['Rank'] == 1].loc[:, 'Unnamed: 0']

    X, y = pd.read_csv("data/X.csv", sep=';', index_col=False), pd.read_csv("data/y.csv", sep=';', index_col=False)
    X_train, y_train, X_test, y_test = split_data_to_pseudonyms(X, y, train_percentage=0.8, random_seed=19)
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
    # analyze_svr_results(args.src_path)
    test_model(args.src_path, 90, 0.7)
