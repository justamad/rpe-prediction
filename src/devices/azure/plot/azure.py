from ..azure import AzureKinect

import pandas as pd
import matplotlib.pyplot as plt

colors = ['red', 'green', 'blue', 'yellow']


def plot_trajectories_for_all_joints(df: pd.DataFrame, file_name: str = None, columns: int = 4):
    """
    Plot the trajectories for the Azure Kinect camera
    @param df: Data Frame that contains the positional or orientation data
    @param file_name: file name of output file
    @param columns: number of columns in the plot
    """
    joints = AzureKinect.get_joints_as_list(df)
    rows, cols = len(joints) // columns, columns
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

        if joint_idx == 31:
            handles, labels = axis.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right')

    fig.suptitle("Azure Kinect")
    fig.tight_layout()
    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()
