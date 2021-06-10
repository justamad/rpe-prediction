import numpy as np
import pandas as pd


def split_data_to_pseudonyms(X: pd.DataFrame, y: pd.DataFrame, train_percentage: float = 0.8,
                             random_seed: bool = False):
    """
    Split the data randomly into train and test sets
    @param X: input data as pandas dataframe
    @param y: ground truth data (multiple columns: groups, rpe, name
    @param train_percentage: percentage value of train subjects
    @param random_seed: flag if a random seed should be used to always generate same split
    @return: tuple with (x_train, y_train, x_test, y_test)
    """
    subject_names = sorted(y['name'].unique())
    nr_subjects = int(len(subject_names) * train_percentage)

    if random_seed:
        np.random.seed(42)

    train_subjects = np.random.choice(subject_names, nr_subjects, replace=False)
    train_indices = y['name'].isin(train_subjects)
    return X.loc[train_indices], y.loc[train_indices], X.loc[~train_indices], y.loc[~train_indices]
