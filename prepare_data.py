from rpe_prediction.config import SubjectDataIterator, FusedAzureLoader, RPELoader
from rpe_prediction.features import calculate_features_sliding_window

import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str, dest='src_path', default="data/processed")
parser.add_argument('--win_size', type=int, dest='win_size', default=90)
parser.add_argument('--overlap', type=float, dest='overlap', default=0.9)
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


def prepare_skeleton_data(iterator, window_size=30, overlap=0.5):
    """
    Prepare Kinect skeleton data using the RawFileIterator
    @param iterator: SubjectDataIterator that delivers all sets over all subjects
    @param window_size: The number of sampled in one window
    @param overlap: The current overlap in percent
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
        features = calculate_features_sliding_window(data.reset_index(), window_size=window_size, overlap=overlap)
        x_data.append(features)

        # Construct y-data with pseudonyms, rpe values and groups
        y = np.repeat([[set_data['subject_name'], set_data['rpe'], set_data['group'], set_data['nr_set']]],
                      len(features), axis=0)
        y_data.append(pd.DataFrame(y, columns=['name', 'rpe', 'group', 'set']))

    return pd.concat(x_data, ignore_index=True), pd.concat(y_data, ignore_index=True)


if __name__ == '__main__':
    file_iterator = SubjectDataIterator(args.src_path).add_loader(RPELoader).add_loader(FusedAzureLoader)
    X, y = prepare_skeleton_data(file_iterator, window_size=args.win_size, overlap=args.overlap)
    X.to_csv("X.csv", index=False, sep=';')
    y.to_csv("y.csv", index=False, sep=';')
    print(X.shape)
    print(y.shape)
