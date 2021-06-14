from rpe_prediction.config import SubjectDataIterator, FusedAzureLoader, RPELoader
from rpe_prediction.models import GridSearching, SVRModelConfig, split_data_to_pseudonyms
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from os.path import join

import numpy as np
import pandas as pd
import prepare_data
import argparse
import logging
import os

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-8s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('my_logger').addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str, dest='src_path', default="data/processed")
parser.add_argument('--out_path', type=str, dest='out_path', default="results")
parser.add_argument('--nr_features', type=int, dest='nr_features', default=40)
args = parser.parse_args()

out_path = join(args.out_path, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(out_path):
    os.makedirs(out_path)

window_sizes = [30, 60, 90]
step_sizes = [10, 15, 20]
file_iterator = SubjectDataIterator(args.src_path).add_loader(RPELoader).add_loader(FusedAzureLoader)

models = [SVRModelConfig()]
logo = LeaveOneGroupOut()

# Iterate over non-sklearn hyperparameters
for window_size in window_sizes:
    for step_size in step_sizes:

        # Generate new data
        X, y = prepare_data.prepare_skeleton_data(file_iterator, window_size=window_size, step_size=step_size)
        X_train, y_train, X_test, y_test = split_data_to_pseudonyms(X, y, train_percentage=0.8, random_seed=True)
        y_train_rpe = y_train['rpe']
        y_train_group = y_train['group']
        y_test_rpe = y_test['rpe']
        y_test_group = y_test['group']

        # Save train and test subjects to file
        np.savetxt(join(out_path, f"train_win_{window_size}_step_{step_size}.txt"), y_train['name'].unique(), fmt='%s')
        np.savetxt(join(out_path, f"test_win_{window_size}_step_{step_size}.txt"), y_test['name'].unique(), fmt='%s')

        # Perform an initial recursive feature elimination
        rfecv = RFECV(SVR(kernel='linear'),
                      min_features_to_select=args.nr_features,
                      step=0.1,
                      n_jobs=-1,
                      verbose=10,
                      cv=logo.get_n_splits(groups=y_train_group))

        # selector = RFE(estimator, n_features_to_select=30, step=20, verbose=10)
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('rfecv', rfecv)
        ])

        pipe.fit(X_train, y_train_rpe)
        logging.info(pipe)
        logging.info(pipe.score(X_test, y_test_rpe))

        # Save RFECV results for later
        rfecv_df = pd.DataFrame(rfecv.ranking_, index=X.columns, columns=['Rank']).sort_values(by='Rank', ascending=True)
        rfecv_df.head()
        rfecv_df.to_csv(join(out_path, f"win_{window_size}_step_{step_size}.csv"))

        # Only use the n most significant features
        X_train = X_train[rfecv.support_]
        X_test = X_test[rfecv.support_]

        # Iterate over models and perform Grid Search
        for model_config in models:
            param_dict = model_config.get_trial_data_dict()
            param_dict['groups'] = y_train_group
            grid_search = GridSearching(**param_dict)
            file_name = join(out_path, f"{str(model_config)}_win_{window_size}_step_{step_size}.csv")
            best_model = grid_search.perform_grid_search(X_train, y_train_rpe, result_file_name=file_name)
            logging.info(best_model.predict(X_test))
            logging.info(best_model.score(X_test, y_test_rpe))
