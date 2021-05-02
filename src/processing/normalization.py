import pandas as pd


def normalize_mean(df: pd.DataFrame, std_dev_factor: float = 1.0):
    """
    Normalize given positional data by subtracting mean and dividing by standard deviation
    @param df: the positional data stored in pandas data frame
    @param std_dev_factor: a factor for the standard deviation
    @return: pandas dataframe with normalized positional data
    """
    return (df - df.mean(axis=0)) / (std_dev_factor * df.std(axis=0))


def normalize_into_interval(df: pd.DataFrame, a: int = 0, b: int = 1):
    """
    Normalize data into given interval [a,b] for each column independently
    @param df: data frame that contains positional or orientation data
    @param a: lower bound of interval
    @param b: upper bound of interval
    @return: normalized data in data frame
    """
    minimum = df.min(axis=0)
    maximum = df.max(axis=0)
    return (b - a) * ((df - minimum) / (maximum - minimum)) + a
