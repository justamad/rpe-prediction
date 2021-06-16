import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def evaluate_for_subject(df):
    data = df[['set', 'prediction']]
    sets = data['set'].unique()

    rpe = df[['rpe', 'set']].drop_duplicates()
    rpe = {k: v for k, v in zip(rpe['set'], rpe['rpe'])}
    means = {}
    stds = {}

    for cur_set in sets:
        mask = data['set'] == cur_set
        cur_data = data.loc[mask]['prediction'].to_numpy()
        mean, std = np.mean(cur_data), np.std(cur_data)
        means[cur_set] = mean
        stds[cur_set] = std

    # create stacked error bars:
    plt.errorbar(means.keys(), means.values(), stds.values(), fmt='ok', lw=1)
    plt.scatter(rpe.keys(), rpe.values(), label="Ground Truth")
    plt.xticks(list(rpe.keys()))
    plt.yticks(np.arange(10, 21))
    plt.xlabel("Set Nr")
    plt.ylabel("RPE value")
    plt.show()
