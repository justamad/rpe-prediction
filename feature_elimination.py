from rpe_prediction.config import SubjectDataIterator, RPELoader, FusedAzureLoader
from rpe_prediction.models import split_data_to_pseudonyms
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import prepare_data
import pandas as pd
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str, dest='src_path', default="data/processed")
parser.add_argument('--out_path', type=str, dest='out_path', default="results")
args = parser.parse_args()

window_sizes = [30, 60, 90]
step_sizes = [5, 10]
file_iterator = SubjectDataIterator(args.src_path).add_loader(RPELoader).add_loader(FusedAzureLoader)

logo = LeaveOneGroupOut()

# Iterate over non-sklearn hyperparameters
for window_size in window_sizes:
    for step_size in step_sizes:
        print("Start trial")

        # Generate new data
        X, y = prepare_data.prepare_skeleton_data(file_iterator, window_size=window_size, step_size=step_size)
        X_train, y_train, X_test, y_test = split_data_to_pseudonyms(X, y, train_percentage=0.8, random_seed=True)
        y_train_rpe = y_train['rpe']
        y_train_group = y_train['group']
        y_test_rpe = y_test['rpe']
        y_test_group = y_test['group']

        rfecv = RFECV(SVR(kernel='linear'),
                      min_features_to_select=40,
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

        rfecv_df = pd.DataFrame(rfecv.ranking_, index=X.columns, columns=['Rank']).sort_values(by='Rank',
                                                                                               ascending=True)

        rfecv_df.head()
        rfecv_df.to_csv(f"win_{window_size}_step_{step_size}.csv", index=False)
        # print(selector.ranking_)
        # print(data.columns[selector.ranking_ == 1])
