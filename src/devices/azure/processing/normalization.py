import pandas as pd


def normalize_mean(df: pd.DataFrame):
    """
    Normalize given positional data by subtracting mean and dividing by standard deviation
    @param df: the positional data stored in pandas data frame
    @return: pandas dataframe with normalized positional data
    """
    return (df - df.mean(axis=0)) / df.std(axis=0)


def normalize_to_range(df: pd.DataFrame, minimum: int = -1, maximum: int = 1):
    min = df.min(axis=0)
    max = df.max(axis=0)
    return (df - min) / (max - min)
