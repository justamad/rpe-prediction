import numpy as np
import pandas as pd

from rpe_prediction.models import GridSearching, SVRModelConfig


def split_data(train_percentage=0.5):
    """
    Split the data randomly into train and test sets
    @param train_percentage: percentage value of train subjects
    @return: tuple with (x_train, y_train, x_test, y_test)
    """
    X = pd.read_csv("../../x.csv", sep=";")
    y = pd.read_csv("../../y.csv", sep=";")
    body_idx_counts = y['group'].unique()
    subjects = int(len(body_idx_counts) * train_percentage)
    train_subjects = np.random.choice(body_idx_counts, subjects, replace=False)
    train_indices = y['group'].isin(train_subjects)
    return X.loc[train_indices], y.loc[train_indices], X.loc[~train_indices], y.loc[~train_indices]


x_train, y_train, x_test, y_test = split_data(train_percentage=0.8)
y_train_rpe = y_train['rpe']
y_train_group = y_train['group']
y_test_rpe = y_test['rpe']
y_test_group = y_test['group']

models = [SVRModelConfig()]

for regression_model_config in models:
    param_dict = regression_model_config.get_trial_data_dict()
    param_dict['groups'] = y_train_group
    grid_search = GridSearching(**param_dict)
    best_model = grid_search.perform_grid_search(x_train, y_train_rpe)
    best_model.predict(x_test)
