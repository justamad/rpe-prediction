from os.path import join

import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str, dest='src_path', default="results/2021-06-14-13-15-57")
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
    print(df)
    print(df.columns)
    df_sum = df.sum(axis=0).sort_values(ascending=True) / len(data)
    df_sum.to_csv(join(input_path, "feature_ranks.csv"))


def analyze_svr_results(input_path):
    """
    Read in results from classifiers
    :param input_path: the path to the result data
    :return:
    """
    files = filter(lambda x: x.startswith('svr') and x.endswith('.csv'), os.listdir(input_path))

    data = []
    for file in files:
        split = file.split('_')
        classifier, win_size, overlap = split[0], int(split[2]), float(split[4][:-4])
        df = pd.read_csv(join(input_path, file), delimiter=';', index_col=False)
        df.insert(0, 'model', classifier)
        df.insert(1, 'win_size', win_size)
        df.insert(2, 'overlap', overlap)
        print(df)
        data.append(df)

    data = pd.concat(data)
    print(data)


if __name__ == '__main__':
    # aggregate_features(args.src_path)
    analyze_svr_results(args.src_path)
