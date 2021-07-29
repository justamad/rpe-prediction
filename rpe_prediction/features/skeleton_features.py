import pandas as pd
import numpy as np

reference = "PELVIS"
joints = [("HIP_RIGHT", "KNEE_RIGHT")]


# ("KNEE_RIGHT","ANKLE_RIGHT")


def calculate_3d_joint_velocities(df):
    diff = (df[0:-1].to_numpy() - df[1:].to_numpy()) ** 2
    grad = np.sqrt(np.sum(diff.reshape(-1, 3), axis=1))
    grad = grad.reshape(diff.shape[0], diff.shape[1] // 3)
    return pd.DataFrame(data=grad, columns=list(map(lambda x: x[:-4], df.columns[::3])))


def calculate_joint_angles(df):
    r = df[[c for c in df.columns if reference in c]].to_numpy()

    for j1, j2 in joints:
        p1 = df[[c for c in df.columns if j1 in c]].to_numpy()
        p2 = df[[c for c in df.columns if j2 in c]].to_numpy()

        v1 = r - p1
        v2 = p2 - r
        angle = (180 / np.pi) * calculate_angle_in_radians_between_vectors(v1, v2)


def calculate_angle_in_radians_between_vectors(v1, v2):
    l_v1 = v1 / np.linalg.norm(v1, axis=1).reshape(-1, 1)
    l_v2 = v2 / np.linalg.norm(v2, axis=1).reshape(-1, 1)
    dot = np.sum(l_v1 * l_v2, axis=1)
    return np.arccos(dot)
