from rpe_prediction.config import SubjectDataIterator, ProcessedLoaderSet
from rpe_prediction.models import GridSearching, SVRModelConfig, split_data_to_pseudonyms
from os.path import join

import prepare_data
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


window_sizes = [30, 60, 90]
step_sizes = [5, 10]
file_iterator = SubjectDataIterator(args.src_path, ProcessedLoaderSet())

models = [SVRModelConfig()]

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

        # Iterate over models and perform Grid Search
        for model_config in models:
            param_dict = model_config.get_trial_data_dict()
            param_dict['groups'] = y_train_group
            grid_search = GridSearching(**param_dict)
            file_name = join(args.out_path, f"{str(model_config)}_winsize_{window_size}_step_{step_size}.csv")
            best_model = grid_search.perform_grid_search(X_train, y_train_rpe, result_file_name=file_name)
            logging.info(best_model.predict(X_test))
            logging.info(best_model.score(X_test, y_test))
