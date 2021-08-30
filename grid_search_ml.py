from rpe_prediction.models import GridSearching, SVRModelConfig, RFModelConfig, split_data_to_pseudonyms, \
    MLPModelConfig, GBRModelConfig, XGBoostRegressor
from rpe_prediction.features import calculate_kinect_feature_set
from rpe_prediction.plot import plot_data_frame_column_wise_as_pdf
from xgboost import XGBRegressor
from sklearn.model_selection import LeaveOneGroupOut
from datetime import datetime
from os.path import join

import numpy as np
import pandas as pd
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

window_sizes = [30, 60, 90, 120]  # 1s, 2s, 3s, 4s
overlaps = [0.5, 0.7, 0.9]

# models = [SVRModelConfig(), GBRModelConfig(), RFModelConfig(), MLPModelConfig()]
models = [XGBoostRegressor()]
logo = LeaveOneGroupOut()

for window_size in window_sizes:
    for overlap in reversed(overlaps):
        # Generate new train and test data
        X, y = calculate_kinect_feature_set(input_path=args.src_path, window_size=window_size, overlap=overlap)
        X_train, y_train, X_test, y_test = split_data_to_pseudonyms(X, y, train_p=0.8, random_seed=42)
        print(f"Shape data: {X.shape}")
        plot_data_frame_column_wise_as_pdf(X, join(out_path, f"importance_winsize_{window_size}_overlap_{overlap}.pdf"))

        # Save train and test subjects to file
        np.savetxt(join(out_path, f"train_win_{window_size}_overlap_{overlap}.txt"), y_train['name'].unique(), fmt='%s')
        np.savetxt(join(out_path, f"test_win_{window_size}_overlap_{overlap}.txt"), y_test['name'].unique(), fmt='%s')

        model = XGBRegressor()
        model.fit(X, y['rpe'])

        df = pd.DataFrame(model.feature_importances_, index=X.columns, columns=["Importance"]) \
            .sort_values(by='Importance', ascending=False)
        df.index.names = ["Feature"]
        df.to_csv(join(out_path, f"importance_winsize_{window_size}_overlap_{overlap}.csv"), sep=';')
        threshold = sorted(model.feature_importances_)[-(args.nr_features + 1)]
        mask = model.feature_importances_ > threshold
        features = list(X.loc[:, mask].columns)
        print(features)

        X_train = X_train.loc[:, mask]
        X_test = X_test.loc[:, mask]
        X_train.to_csv(join(out_path, f"X_train_win_{window_size}_overlap_{overlap}.csv"), index=False, sep=';')
        X_test.to_csv(join(out_path, f"X_test_win_{window_size}_overlap_{overlap}.csv"), index=False, sep=';')

        for model_config in models:
            param_dict = model_config.get_trial_data_dict()
            grid_search = GridSearching(groups=y_train['group'], **param_dict)
            file_name = join(out_path, f"{str(model_config)}_win_{window_size}_overlap_{overlap}.csv")
            best_model = grid_search.perform_grid_search(X_train, y_train['rpe'], output_file=file_name)
            logging.info(best_model.predict(X_test))
            logging.info(best_model.score(X_test, y_test['rpe']))


def feature_elimination():
    pass
    # selector = SelectFromModel(estimator=LogisticRegression()).fit(X, y['rpe'])
    # print(selector.estimator_.coef_)
    # print(selector.threshold_)
    # print(selector.get_support())
    # features = list(X.loc[:, selector.get_support()].columns)
    # print(features)

    # clf = ExtraTreesClassifier(n_estimators=50)
    # clf = clf.fit(X, y)
    # print(clf.feature_importances_)
    # # Perform an initial recursive feature elimination
    # rfe = RFECV(SVR(kernel='linear'),
    #             min_features_to_select=args.nr_features,
    #             step=0.1,
    #             n_jobs=-1,
    #             verbose=10,
    #             cv=logo.get_n_splits(groups=y_train['group']))
    #
    #
    # pipe = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('rfe', rfe)
    # ])
    # pipe.fit(X_train, y_train['rpe'])
    #
    # # Save RFECV results for later
    # rfe_df = pd.DataFrame(rfe.ranking_, index=X.columns, columns=['Rank']).sort_values(by='Rank', ascending=True)
    # rfe_df.index.names = ["Feature"]
    # rfe_df.to_csv(join(out_path, f"features_win_{window_size}_overlap_{overlap}.csv"), sep=';')
