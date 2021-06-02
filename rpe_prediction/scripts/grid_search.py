from rpe_prediction.models import GridSearching, SVRModelConfig

import pandas as pd
import numpy as np


def split_data(train_percentage=0.5):
    X = pd.read_csv("../../x.csv", sep=";")
    y = pd.read_csv("../../y.csv", sep=";")
    body_idx_counts = y['group'].unique()
    subjects = int(len(body_idx_counts) * train_percentage)
    test_subjects = np.random.choice(body_idx_counts, subjects)
    return


split_data(train_percentage=0.8)

models = [SVRModelConfig()]


# Main Loop: iterate over learning models
for regression_model_config in models:
    param_dict = regression_model_config.get_trial_data_dict()
    param_dict['groups'] = groups
    grid_search = GridSearching(**param_dict)
    best_model = grid_search.perform_grid_search(X, y)
    best_model.predict()
