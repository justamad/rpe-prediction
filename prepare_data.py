import argparse

import numpy as np
import pandas as pd

from rpe_prediction.config import SubjectDataIterator, ProcessedLoaderSet
from rpe_prediction.features import calculate_features_sliding_window

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str, dest='src_path', default="data/processed")
args = parser.parse_args()


def prepare_skeleton_data(iterator, window_size=30, step_size=2):
    """
    Prepare Kinect skeleton data using the RawFileIterator
    @param iterator: SubjectDataIterator that delivers all sets over all subjects
    @param window_size: The number of sampled in one window
    @param step_size: the step size the window is moved over time series
    @return: Tuple that contains input data and labels (input, labels)
    """
    x_data = []
    y_labels = []
    groups = []
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
    std_dev = {k: pd.concat(v, ignore_index=True).std(axis=0) for k, v in trials.items()}

    for set_data in iterator.iterate_over_all_subjects():
        subject_name = set_data['subject_name']
        data = pd.read_csv(set_data['azure'], sep=';', index_col=False).set_index('timestamp')

        # normalize it using the pre-calculated values
        data = (data - means[subject_name]) / std_dev[subject_name]

        features = calculate_features_sliding_window(data.reset_index(), window_size=window_size, step_size=step_size)
        x_data.append(features)
        y_labels.extend([set_data['rpe'] for _ in range(len(features))])
        groups.extend([set_data['group'] for _ in range(len(features))])

    X = pd.concat(x_data, ignore_index=True)
    y = pd.DataFrame(np.stack([y_labels, groups], axis=1), columns=["rpe", "group"])
    return X, y


if __name__ == '__main__':
    file_iterator = SubjectDataIterator(args.src_path, ProcessedLoaderSet())
    X, y = prepare_skeleton_data(file_iterator, window_size=60, step_size=5)

    X.to_csv("x.csv", index=False, sep=';')
    y.to_csv("y.csv", index=False, sep=';')
    print(X.shape)
    print(y.shape)
