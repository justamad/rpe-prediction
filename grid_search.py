from rpe_prediction.config import SubjectDataIterator, ProcessedLoaderSet
from rpe_prediction.models import GridSearching, SVRModelConfig
from os.path import join

import prepare_data
import numpy as np
import pandas as pd
import argparse
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-8s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('my_logger').addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str, dest='src_path', default="data/processed")
parser.add_argument('--out_path', type=str, dest='out_path', default="results")
args = parser.parse_args()


def split_data(X, y, train_percentage=0.5):
    """
    Split the data randomly into train and test sets
    @param train_percentage: percentage value of train subjects
    @return: tuple with (x_train, y_train, x_test, y_test)
    """
    body_idx_counts = y['group'].unique()
    subjects = int(len(body_idx_counts) * train_percentage)
    train_subjects = np.random.choice(body_idx_counts, subjects, replace=False)
    train_indices = y['group'].isin(train_subjects)
    return X.loc[train_indices], y.loc[train_indices], X.loc[~train_indices], y.loc[~train_indices]


window_sizes = [30, 60, 90]
step_sizes = [5, 10]
file_iterator = SubjectDataIterator(args.src_path, ProcessedLoaderSet())

models = [SVRModelConfig()]

# Iterate over non-sklearn hyperparameters
for window_size in window_sizes:
    for step_size in step_sizes:

        # Generate new data
        X, y = prepare_data.prepare_skeleton_data(file_iterator, window_size=window_size, step_size=step_size)
        labels = y['rpe']
        groups = y['group']

        for model_config in models:

            param_dict = model_config.get_trial_data_dict()
            param_dict['groups'] = groups
            grid_search = GridSearching(**param_dict)
            file_name = join(args.out_path, f"{str(model_config)}_winsize_{window_size}_step_{step_size}.csv")
            best_model = grid_search.perform_grid_search(X, labels, result_file_name=file_name)
            # best_model.predict(x_test)

# x_train, y_train, x_test, y_test = split_data(X, y, train_percentage=0.8)
# y_train_rpe = y_train['rpe']
# y_train_group = y_train['group']
# y_test_rpe = y_test['rpe']
# y_test_group = y_test['group']
