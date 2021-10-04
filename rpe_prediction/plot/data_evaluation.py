from scipy.stats import pearsonr, spearmanr
from matplotlib import ticker
from pandas.api.types import is_numeric_dtype
from rpe_prediction.processing import get_hsv_color

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


def plot_time_series_predictions(df: pd.DataFrame, file_name: str = None):
    plt.plot(df['rpe'], label="Ground Truth")
    plt.plot(df['prediction'], label="Predictions")
    plt.yticks(np.arange(10, 21))
    # plt.yticks(np.arange(0, 1.1, 0.1))

    plt.xlabel("Frames (Windows)")
    plt.ylabel("RPE value")
    # plt.title(f'')

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()

    plt.clf()
    plt.cla()
    plt.close()


def plot_parallel_coordinates(df: pd.DataFrame,
                              color_column: str,
                              columns: list = None,
                              title: str = None,
                              param_prefix: str = None,
                              file_name: str = None):
    if columns is not None:
        df = df[columns]
    else:
        df = df[[c for c in df.columns if "param" in c or "mean_test" in c or "std_test" in c]]

    if param_prefix is not None:
        df.columns = df.columns.str.replace(param_prefix, '')

    df = df.fillna(0)
    columns = df.columns
    trials, parameters = df.shape
    x = list(range(parameters))
    fig, axes = plt.subplots(1, len(x) - 1, sharey=False, figsize=(20, 15))

    # Get min, max and range for each column - Normalize the data for each column
    min_max_range = {}
    labels = {}
    for col in df.columns:
        if not is_numeric_dtype(df[col]):
            values = df[col].unique()
            mapping = {v: k for k, v in enumerate(values)}
            df = df.replace({col: mapping})
            labels[col] = values

        min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
        df.loc[:, col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))

    # Plot each row / trial
    for i, ax in enumerate(axes):
        for idx in df.index:
            ax.plot(x, df.loc[idx, columns], color=get_hsv_color(df.loc[idx, color_column], 1))
        ax.set_xlim([x[i], x[i + 1]])

    def set_ticks_for_axis(dim_id: int, axis, ticks: int = 6):
        # Tick positions based on normalised data, tick labels are based on original data
        min_val, max_val, val_range = min_max_range[columns[dim_id]]

        if columns[dim_id] in labels:
            tick_labels = labels[columns[dim_id]]
            ticks = len(tick_labels)
        else:
            step = val_range / float(ticks - 1)
            tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]

        norm_min = df[columns[dim_id]].min()
        norm_range = np.ptp(df[columns[dim_id]])
        norm_step = norm_range / float(ticks - 1)
        ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
        axis.yaxis.set_ticks(ticks)
        axis.set_yticklabels(tick_labels)

    for dim, ax in enumerate(axes):
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        set_ticks_for_axis(dim, ax)
        ax.set_xticklabels([columns[dim]])
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

    # Move the final axis' ticks to the right-hand side
    ax = plt.twinx(axes[-1])
    ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    set_ticks_for_axis(dim_id=len(axes), axis=ax)
    ax.set_xticklabels([columns[-2], columns[-1]])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    # Remove space between subplots
    plt.subplots_adjust(wspace=0)

    if title is not None:
        plt.suptitle(title)

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()

    plt.clf()
    plt.cla()
    plt.close()
