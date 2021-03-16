import pandas as pd
import matplotlib.pyplot as plt


def plot_trajectories_for_all_joints(azure_cam: pd.DataFrame, file_name: str = None, columns: int = 4):
    """
    Plot the trajectories for the Azure Kinect camera
    @param azure_cam: camera object
    @param file_name: file name of output file
    @param columns: number of columns in the plot
    """
    joints = azure_cam.get_joints_as_list()

    rows, cols = len(joints) // columns, columns
    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))

    for joint_idx, joint in enumerate(joints):
        joint_data = azure_cam[joint].to_numpy()
        axis = axs[joint_idx // columns, joint_idx % columns]
        axis.set_title(joint)
        # Plot with color coding
        for idx, (ax, color) in enumerate(zip(['x', 'y', 'z'], ['red', 'green', 'blue'])):
            axis.plot(joint_data[:, idx], color=color, label=ax)

        if joint_idx == 31:
            handles, labels = axis.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right')

    fig.suptitle("Azure Kinect")
    fig.tight_layout()
    plt.savefig(file_name)
