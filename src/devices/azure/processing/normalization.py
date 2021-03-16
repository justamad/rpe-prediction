import pandas as pd


def normalize_mean(df: pd.DataFrame):
    """
    Normalize given positional data by subtracting mean and dividing by standard deviation
    @param df: the positional data stored in pandas data frame
    @return: pandas dataframe with normalized positional data
    """
    return (df - df.mean(axis=0)) / df.std(axis=0)
