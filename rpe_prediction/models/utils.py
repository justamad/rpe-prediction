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


def normalize_rpe_values(df_ground_truth):
    """
    Normalize the RPE values according to Pernek et al. by applying min-max scaling
    :param df_ground_truth: data frame that contains y-output format
    :return: data frame with normalized RPE values per subject
    """
    subjects = df_ground_truth['group'].unique()
    for subject_id in subjects:
        mask = df_ground_truth['group'] == subject_id
        df_subject = df_ground_truth[mask]
        mini, maxi = df_subject['rpe'].min(), df_subject['rpe'].max()
        df_ground_truth.loc[mask, 'rpe'] = (df_subject['rpe'] - mini) / (maxi - mini)

    return df_ground_truth


if __name__ == '__main__':
    X = pd.read_csv("../../data/X.csv", sep=';', index_col=False)
    y = pd.read_csv("../../data/y.csv", sep=';', index_col=False)
    y = normalize_rpe_values(y)
    print(y)
