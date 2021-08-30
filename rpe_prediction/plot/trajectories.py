from .pdf_writer import PDFWriter
from rpe_prediction.processing import get_joint_names_from_columns_as_list

import matplotlib.pyplot as plt
import pandas as pd
import math

colors = ['red', 'green', 'blue', 'yellow']


def plot_sensor_data_for_axes(df: pd.DataFrame, title: str, joints: list, file_name: str = None, columns: int = 4):
    """
    Plots the trajectories for the Azure Kinect camera
    @param df: Data Frame that contains the positional or orientation data
    @param title: The title of the graph
    @param joints: A list of the current joints within the data frame
    @param file_name: file name of output file
    @param columns: number of columns in the plot
    """
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
    """
    Plots trajectories or other sensor data for a given data frame. In each subplot only one axis is shown
    @param df: data frame that contains the sensor data
    @param title: title of the final plot
    @param file_name: file name in case plot should be saved to disk
    @param columns: number of columns for sub plots
    """
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
