from src.processing import get_joint_names_from_columns_as_list
from .pdf_writer import PDFWriter
from scipy.stats import norm

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

