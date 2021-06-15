from rpe_prediction.config import SubjectDataIterator, FusedAzureLoader, RPELoader
from rpe_prediction.models import split_data_to_pseudonyms
from sklearn.svm import SVR
from os.path import join

import pandas as pd
import os
import argparse
import prepare_data

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
    print(features)

    file_iterator = SubjectDataIterator(args.data_path).add_loader(RPELoader).add_loader(FusedAzureLoader)
    X, y = prepare_data.prepare_skeleton_data(file_iterator, window_size=win_size, overlap=overlap)
    X_train, y_train, X_test, y_test = split_data_to_pseudonyms(X, y, train_percentage=0.8, random_seed=True)

    model = SVR(kernel='linear', gamma=0.001, C=1.0)



if __name__ == '__main__':
    # aggregate_features(args.src_path)
    # analyze_svr_results(args.src_path)
    test_model(args.src_path, 90, 0.7)
