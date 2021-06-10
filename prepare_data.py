from rpe_prediction.config import SubjectDataIterator, ProcessedLoaderSet
from rpe_prediction.features import calculate_features_sliding_window

import argparse
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-8s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('my_logger').addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str, dest='src_path', default="data/processed")
args = parser.parse_args()


def calculate_means_over_subjects(iterator):
    """
    Calculate mean and standard deviations over individual subjects
    @param iterator: RawDataIterator
    @return: A tuple of dictionaries that map subjects to mean and std devs: ({S1: mean1, ...}, {S1: std_dev1, ...})
    """
    trials = {}

    # Iterate over all trials to determine mean and std dev
    for set_data in iterator.iterate_over_all_subjects():
        subject_name = set_data['subject_name']
        data = pd.read_csv(set_data['azure'], sep=';', index_col=False).set_index('timestamp')
        if subject_name not in trials:
            trials[subject_name] = [data]
        else:
            trials[subject_name].append(data)

    means = {k: pd.concat(v, ignore_index=True).mean(axis=0) for k, v in trials.items()}
    std_devs = {k: pd.concat(v, ignore_index=True).std(axis=0) for k, v in trials.items()}
    return means, std_devs


def prepare_skeleton_data(iterator, window_size=30, step_size=2):
    """
    Prepare Kinect skeleton data using the RawFileIterator
    @param iterator: SubjectDataIterator that delivers all sets over all subjects
    @param window_size: The number of sampled in one window
    @param step_size: the step size the window is moved over time series
    @return: Tuple that contains input data and labels (input, labels)
    """
    means, std_dev = calculate_means_over_subjects(iterator)
    x_data = []
    y_data = []

    for set_data in iterator.iterate_over_all_subjects():
        subject_name = set_data['subject_name']
        data = pd.read_csv(set_data['azure'], sep=';', index_col=False).set_index('timestamp')

        # Normalize dataframe using the pre-calculated values
        data = (data - means[subject_name]) / std_dev[subject_name]
        features = calculate_features_sliding_window(data.reset_index(), window_size=window_size, step_size=step_size)
        x_data.append(features)

        # Construct y-data with pseudonyms, rpe values and groups
        y = np.repeat([[set_data['subject_name'], set_data['rpe'], set_data['group']]], len(data), axis=0)
        y_data.append(pd.DataFrame(y, columns=['name', 'rpe', 'group']))

    return pd.concat(x_data, ignore_index=True), pd.concat(y_data, ignore_index=True)


if __name__ == '__main__':
    file_iterator = SubjectDataIterator(args.src_path, ProcessedLoaderSet())
    X, y = prepare_skeleton_data(file_iterator, window_size=60, step_size=5)

    X.to_csv("x.csv", index=False, sep=';')
    y.to_csv("y.csv", index=False, sep=';')
    print(X.shape)
    print(y.shape)
