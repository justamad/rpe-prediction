from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def evaluate_for_subject(df):
    """
    Plot result data for subject
    :param df: dataframe that holds predictions and ground truth values
    :return: None
    """
    data = df[['set', 'prediction']]
    sets = data['set'].unique()
    rpe = df[['rpe', 'set']].drop_duplicates()

    rpe = {cur_set: cur_rpe for cur_set, cur_rpe in zip(rpe['set'], rpe['rpe'])}
    means = {}
    stds = {}

    for cur_set in sets:
        mask = data['set'] == cur_set
        cur_data = data.loc[mask]['prediction']
        means[cur_set] = np.mean(cur_data)
        stds[cur_set] = np.std(cur_data)

    pear, p = pearsonr([means[i] for i in sets], [rpe[i] for i in sets])

    # create stacked error bars:
    plt.errorbar(means.keys(), means.values(), stds.values(), fmt='ok', lw=1)
    plt.scatter(rpe.keys(), rpe.values(), label="Ground Truth")
    plt.xticks(list(rpe.keys()))
    # plt.yticks(np.arange(10, 21))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel("Set Nr")
    plt.ylabel("RPE value")
    plt.title(f'Correlation Pearson: {pear:.2f}')
    plt.show()
