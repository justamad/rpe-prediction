import pandas as pd
import numpy as np


def calculate_magnitude(df, axis_suffix=' (x)'):
    """
    Calculate the magnitude for a given data frame
    @param df: data frame that contains sensor data each with an (x,y,z) axis
    @param axis_suffix: the axis suffix in the file
    @return: the magnitude for each joint/sensor in a data frame
    """
    joints = get_joints_as_list(df, axis_suffix)
    result = []
    for joint in joints:
        data = df[[c for c in df.columns if joint in c]]
        result.append(np.sqrt(np.square(data).sum(axis=1)))

    result = pd.concat(result, axis=1)
    result.columns = joints
    return result


def calculate_and_append_magnitude(df, column_prefix="ACCELERATION"):
    data = df[[c for c in df.columns if column_prefix in c]]
    magnitude = np.sqrt(np.square(data).sum(axis=1))
    df['MAGNITUDE'] = magnitude
    return df


def calculate_gradient(df):
    """
    Calculate the first gradient using central differences in given data frame
    :param df: origin data frame that contains time series data
    :return: data frame that contains gradients
    """
    data = df.to_numpy()
    grad = np.gradient(data, axis=0)
    return pd.DataFrame(grad, columns=df.columns)


def calculate_second_gradient(df):
    """
    Calculate a second gradient using central differences for data in given data frame
    :param df: origin data frame
    :return: data frame that contains second gradient over rows
    """
    data = df.to_numpy()
    second_grad = np.gradient(np.gradient(data, axis=0), axis=0)
    return pd.DataFrame(second_grad, columns=df.columns)


def filter_dataframe(df: pd.DataFrame, excluded_matches: list):
    """
    Filters out columns in dataframe that contain partial string given in the list
    :param df: the origin data frame
    :param excluded_matches: a list with substrings that should be filtered out
    :return: data frame with filtered columns
    """
    for excluded_part in excluded_matches:
        df = df.loc[:, ~df.columns.str.contains(excluded_part)]
    return df


def reshape_data_for_ts(df, joints):
    """
    Reshape data to the specific format for the tsfresh library
    :param df: current dataframe
    :param joints: list of all joints in dataframe
    :return: reshaped data frame
    """
    data_result = []
    for joint in joints:
        columns = ['timestamp'] + [c for c in df.columns if joint in c]
        data = df[columns].copy()
        data.columns = [c.replace(joint, '') for c in data.columns]
        data['id'] = joint
        data_result.append(data)

    return pd.concat(data_result, ignore_index=True)


def reshape_data_from_ts(df):
    df2 = df.reset_index().melt(id_vars=['index'], value_vars=df.columns)
    df2['name'] = df2['variable'] + '_' + df2['index']
    df2 = df2[['name', 'value']].set_index('name').transpose()
    return df2


def get_joints_as_list(df, joints):
    """
    Identify all joints given in the data frame, remove all axes or types from columns
    @param df: a pandas data frame with sensor data
    @param joints: a full list of joints which forms a superset of possible joints in data frame
    @return: list of filtered joint names in alphabetical order
    """
    left_joints = list(filter(lambda x: any([c for c in df.columns if x in c]), joints))
    print(left_joints)
    return left_joints
