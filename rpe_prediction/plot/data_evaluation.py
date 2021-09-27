from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_rpe_predictions_from_dataframe(df: pd.DataFrame, file_name: str = None):
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

    # Create stacked error bars:
    plt.errorbar(means.keys(), means.values(), stds.values(), fmt='ok', lw=1, ecolor='green', mfc='green')
    plt.scatter(rpe.keys(), rpe.values(), label="Ground Truth", c='red')
    plt.xticks(list(rpe.keys()))
    plt.yticks(np.arange(10, 21))
    # plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel("Set Nr")
    plt.ylabel("RPE value")
    plt.title(f'Correlation Pearson: {pear:.2f}')

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()

    plt.clf()
    plt.cla()
    plt.close()
