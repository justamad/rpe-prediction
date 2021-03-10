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
gaitup = GaitUp(join(args.src_path, "gaitup"))

# Process individual sets
for counter, sensor_trial in enumerate(config.iterate_over_trials()):
    print(f"Convert trial {counter}...")
    azure = AzureKinect(join(args.src_path, "azure", f"0{counter + 1}_sub", "positions_3d.csv"))
    faros = Faros(join(args.src_path, "faros"), *sensor_trial['faros'])
    gaitup_trial = gaitup.cut_data(*sensor_trial['gaitup'])  # Map indices to according sampling frequency

    # Synchronize signals
    report_path = join(args.report_path, f"{counter}_azure")
    clean_up_dirs(report_path)
    gaitup_clock = synchronize_signals(azure, gaitup_trial, method="correlation")  # , show=True, path=report_path)
    faros_clock = synchronize_signals(azure, faros, method="correlation")  # , show=True, path=report_path)
    print(len(azure.data), len(gaitup_trial.data))
