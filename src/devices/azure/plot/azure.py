import matplotlib.pyplot as plt


# def plot_trajectories_for_all_joints(azure_cam):
#     joints = azure_cam.get_joints_as_list()
#
#     rows, cols = len(joints), 3
#     print(joints)
#     print(rows, cols)
#     fig, axs = plt.subplots(rows, cols)
#
#     for joint_idx, joint in enumerate(joints):
#         joint_data = azure_cam[joint].to_numpy()
#         print(joint_data.shape)
#         for axis_idx in range(3):
#             # print(axis)
#             axs[joint_idx, axis_idx].plot(joint_data[:, axis_idx])
#
#     plt.show()


def plot_trajectories_for_all_joints(azure_cam, file_name, columns=4):
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
    # plt.show()
