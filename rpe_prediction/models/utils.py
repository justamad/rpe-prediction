import numpy as np
import pandas as pd


def split_data_to_pseudonyms(x: pd.DataFrame, y: pd.DataFrame, train_percentage: float = 0.8,
                             random_seed: int = None):
    """
    Split the data randomly into train and test sets
    @param x: input data as pandas dataframe
    @param y: ground truth data (multiple columns: groups, rpe, name)
    @param train_percentage: percentage value of train subjects
    @param random_seed: a random seed as integer, if not set, use a random split
    @return: tuple with four pandas data frames (x_train, y_train, x_test, y_test)
    """
    subject_names = sorted(y['name'].unique())
    nr_subjects = int(len(subject_names) * train_percentage)

    if random_seed is not None:
        np.random.seed(random_seed)

    train_subjects = np.random.choice(subject_names, nr_subjects, replace=False)
    train_indices = y['name'].isin(train_subjects)
    return x.loc[train_indices].copy(), y.loc[train_indices].copy(), \
           x.loc[~train_indices].copy(), y.loc[~train_indices].copy()
