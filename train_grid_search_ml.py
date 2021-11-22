from src.features import calculate_kinect_feature_set
from src.utils import create_folder_if_not_already_exists
from src.plot import plot_feature_distribution_as_pdf, plot_feature_correlation_heatmap
from datetime import datetime
from os.path import join, isfile
from argparse import ArgumentParser

from src.ml import (
    GridSearching,
    split_data_based_on_pseudonyms,
    normalize_features_z_score,
    normalize_rpe_values_min_max,
    eliminate_features_with_xgboost_coefficients,
    eliminate_features_with_rfe,
    MLPModelConfig,
    GBRModelConfig,
)

import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-8s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M:%S',
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('my_logger').addHandler(console)

parser = ArgumentParser()
parser.add_argument('--src_path', type=str, dest='src_path', default="data/processed")
parser.add_argument('--feature_path', type=str, dest='feature_path', default="data/features")
parser.add_argument('--result_path', type=str, dest='result_path', default="results")
parser.add_argument('--nr_features', type=int, dest='nr_features', default=100)
parser.add_argument('--nr_augment', type=int, dest='nr_augment', default=0)
parser.add_argument('--borg_scale', type=int, dest='borg_scale', default=5)
args = parser.parse_args()


if __name__ == '__main__':
    models = [MLPModelConfig(), GBRModelConfig()]

    result_path = join(args.result_path, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    for model in models:
        create_folder_if_not_already_exists(join(result_path, str(model)))
    create_folder_if_not_already_exists(args.feature_path)

    window_sizes = [30]  # , 60, 90, 120]  # 1s, 2s, 3s, 4s
    overlaps = [0.5]  # , 0.7, 0.9]

    for win_size in window_sizes:
        for overlap in reversed(overlaps):

            # Cache pre-calculated features
            X_file = join(args.feature_path, f"X_win_{win_size}_overlap_{overlap}_augmentation_{args.nr_augment}.csv")
            y_file = join(args.feature_path, f"y_win_{win_size}_overlap_{overlap}_augmentation_{args.nr_augment}.csv")

            if isfile(X_file) and isfile(y_file):
                X_orig = pd.read_csv(X_file, sep=';', index_col=False)
                y_orig = pd.read_csv(y_file, sep=';', index_col=False)
            else:
                X_orig, y_orig = calculate_kinect_feature_set(
                    input_path=args.src_path,
                    statistical_features=True,
                    window_size=win_size,
                    overlap=overlap,
                    nr_augmentation_iterations=args.nr_augment
                )

                X_orig.to_csv(X_file, sep=';', index=False)
                y_orig.to_csv(y_file, sep=';', index=False)

            # features = X_orig.filter(regex="3D_VELOCITY")
            # features = features.filter(regex="maximum")
            # plot_feature_correlation_heatmap(features)

            X_scaled = normalize_features_z_score(X_orig)
            y_norm = normalize_rpe_values_min_max(df=y_orig, digitize=True, bins=args.borg_scale)
            # plot_feature_distribution_as_pdf(X_orig, X_scaled,
            #                                  join(result_path, f"features_win_{win_size}_overlap_{overlap}.pdf"))

            X_train, y_train, X_test, y_test = split_data_based_on_pseudonyms(
                X=X_scaled,
                y=y_orig,
                train_p=0.6,
                random_seed=42,
            )

            # X_train, X_test = eliminate_features_with_xgboost_coefficients(
            #     X_train=X_train,
            #     y_train=y_train['rpe'],
            #     X_test=X_test,
            #     y_test=y_test['rpe'],
            #     analyze_features=False,
            #     win_size=win_size,
            #     overlap=overlap,
            #     nr_features=args.nr_features,
            #     path=result_path
            # )

            X_train, X_test = eliminate_features_with_rfe(
                X_train=X_train,
                y_train=y_train['rpe'],
                X_test=X_test,
                y_test=y_test['rpe'],
                window_size=win_size,
                path=result_path,
                step=100,
            )

            # Save train and test subjects to file
            np.savetxt(join(result_path, f"train_win_{win_size}_overlap_{overlap}.txt"), y_train['name'].unique(), fmt='%s')
            np.savetxt(join(result_path, f"test_win_{win_size}_overlap_{overlap}.txt"), y_test['name'].unique(), fmt='%s')

            X_train.to_csv(join(result_path, f"X_train_win_{win_size}_overlap_{overlap}.csv"), index=False, sep=';')
            y_train.to_csv(join(result_path, f"y_train_win_{win_size}_overlap_{overlap}.csv"), index=False, sep=';')

            X_test.to_csv(join(result_path, f"X_test_win_{win_size}_overlap_{overlap}.csv"), index=False, sep=';')
            y_test.to_csv(join(result_path, f"y_test_win_{win_size}_overlap_{overlap}.csv"), index=False, sep=';')

            for model_config in models:
                param_dict = model_config.get_trial_data_dict()
                grid_search = GridSearching(groups=y_train['group'], **param_dict)
                file_name = join(result_path, str(model_config), f"win_{win_size}_overlap_{overlap}.csv")
                best_model = grid_search.perform_grid_search_with_cv(X_train, y_train['rpe'], output_file=file_name)
                logging.info(best_model.predict(X_test))
                logging.info(best_model.score(X_test, y_test['rpe']))
