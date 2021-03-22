import pandas as pd
import numpy as np


def calculate_magnitude(df):
    """
    Calculate the magnitude for a given data frame
    @param df: data frame that contains sensor data each with an (x,y,z) axis
    @return: the magnitude for each joint/sensor in a data frame
    """
    joints = get_joints_as_list(df)
    result = []
    for joint in joints:
        joint_data = df[[c for c in df.columns if joint.lower() in c]].to_numpy()
        magnitude = np.sqrt(np.sum(np.square(joint_data), axis=1))
        result.append(magnitude)

    data = np.array(result).reshape(-1, len(joints))
    return pd.DataFrame(data=data, columns=joints)


def get_joints_as_list(df):
    """
    Identify all joints given in the data frame, remove all axes or types from columns
    @param df: a pandas data frame with sensor data
    @return: list of filtered joint names
    """
    return list(set([c[:-4] for c in df.columns]))
