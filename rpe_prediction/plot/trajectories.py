from rpe_prediction.processing import get_joint_names_from_columns_as_list, get_hsv_color
from .pdf_writer import PDFWriter
from scipy.stats import norm
from matplotlib import ticker
from pandas.api.types import is_numeric_dtype

import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import math

colors = ['red', 'green', 'blue', 'yellow']


def plot_sensor_data_for_axes(df: pd.DataFrame, title: str, joints: list, file_name: str = None, columns: int = 4):
    joints = get_joint_names_from_columns_as_list(df, joints)
    rows, cols = math.ceil(len(joints) / columns), columns
    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))

    for joint_idx, joint in enumerate(joints):
        joint_data = df[[c for c in df.columns if joint.lower() in c]]
        axes = [c[-2:-1] for c in joint_data.columns]
        joint_data = joint_data.to_numpy()
        axis = axs[joint_idx // columns, joint_idx % columns]
        axis.set_title(joint.replace('_', ' ').title())

        # Plot with color coding
        for idx, (ax, color) in enumerate(zip(axes, colors)):
            axis.plot(joint_data[:, idx], color=color, label=ax)

        if joint_idx == len(joints) - 1:
            handles, labels = axis.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right')

    fig.suptitle(title)
    fig.tight_layout()
    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()

    plt.close()
    plt.cla()
    plt.clf()


def plot_sensor_data_for_single_axis(df: pd.DataFrame, title: str, file_name: str = None, columns: int = 4):
    rows, cols = math.ceil(len(df.columns) / columns), columns
    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))

    for joint_idx, joint in enumerate(df.columns):
        joint_data = df[[c for c in df.columns if joint.lower() in c]].to_numpy()
        axis = axs[joint_idx // columns, joint_idx % columns]
        axis.set_title(joint.replace('_', ' ').title())
        axis.plot(joint_data)

    fig.suptitle(title)
    fig.tight_layout()
    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()


def plot_data_frame_column_wise_as_pdf(df: pd.DataFrame, file_name: str):
    pp = PDFWriter(file_name)
    for column in df:
        plt.close()
        plt.figure()
        plt.title(column)
        plt.plot(df[column])
        pp.save_figure()
        plt.clf()

    pp.close_and_save_file(add_bookmarks=False)


def plot_feature_distribution_as_pdf(df: pd.DataFrame, df_norm: pd.DataFrame, file_name: str):
    pp = PDFWriter(file_name)

    for column in df:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 15))
        data = df[column]
        ax1.plot(data)
        ax1.set_title("Raw Feature")

        # Plot distribution
        mu, std = norm.fit(data)
        ax2.hist(data, bins=100, density=True, alpha=0.6, color='g')
        x = np.linspace(min(data), max(data), 100)
        p = norm.pdf(x, mu, std)
        ax2.plot(x, p, 'k', linewidth=2)
        ax2.set_title(f"Gaussian Fit: mu={mu:.2f}, std={std:.2f}")

        ax3.plot(df_norm[column])
        ax3.set_title("Normalized Feature")

        fig.suptitle(f"{column}")
        pp.save_figure()
        ax1.cla()
        ax2.cla()
        ax3.cla()
        plt.close()
        fig.clf()

    pp.close_and_save_file(add_bookmarks=False)


def plot_feature_correlation_heatmap(df: pd.DataFrame):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(50, 50))
    sb.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sb.diverging_palette(20, 220, s=100, l=30, n=200),
        square=True,
        annot=True,
        # annot_kws={"size": 40 / np.sqrt(len(corr))},
        # linewidths=0.5,
        # cbar_kws={"shrink": .5}
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=90,
        horizontalalignment='right'
    )

    plt.tight_layout()
    plt.show()


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
    fig, axes = plt.subplots(1, len(x) - 1, sharey=False, figsize=(15, 5))

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
        # Tick positions based on normalised data
        # Tick labels are based on original data
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

    # Move the final axis' ticks to the right-hand side
    ax = plt.twinx(axes[-1])
    ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    set_ticks_for_axis(dim_id=len(axes), axis=ax)
    ax.set_xticklabels([columns[-2], columns[-1]])

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
