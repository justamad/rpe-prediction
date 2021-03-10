from src.faros import Faros
from src.azure import AzureKinect
from src.gaitup import GaitUp
from src.processing import synchronize_signals
from src.config import ConfigReader
from os.path import join

import shutil
import os
import argparse

# Define Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str, dest='src_path', default="data/bjarne_trial")
parser.add_argument('--report_path', type=str, dest='report_path', default="reports")
args = parser.parse_args()


def clean_up_dirs(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


# Load data
config = ConfigReader(join(args.src_path, "config.json"))
# faros = Faros(join(args.src_path, "faros"))
gaitup = GaitUp(join(args.src_path, "gaitup"))

# Process individual sets
for counter, sensor_trial in enumerate(config.iterate_over_trials()):
    print(f"Convert trial {counter}...")
    # faros_set = faros.cut_trial(*sensor_trial['faros'])

    # Map indices to according sampling frequency
    start_idx, end_idx = sensor_trial['gaitup']
    # start_idx = int(start_idx * (100 / 128))
    # end_idx = int(end_idx * (100 / 128))
    gaitup_trial = gaitup.cut_data(start_idx, end_idx)
    azure = AzureKinect(join(args.src_path, "azure",  f"0{counter + 1}_sub", "positions_3d.csv"))

    # Synchronize signals
    report_path = join(args.report_path, f"{counter}_azure")
    clean_up_dirs(report_path)
    azure, gaitup_trial = synchronize_signals(azure, gaitup_trial, method="correlation")  # , show=True, path=report_path)
    print(len(azure.data), len(gaitup_trial.data))
