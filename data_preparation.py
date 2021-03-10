from src.devices import AzureKinect, Faros, GaitUp
from src.processing import synchronize_signals, normalize_signal
from src.config import ConfigReader
from os.path import join

import shutil
import os
import argparse
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Define Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str, dest='src_path', default="data/bjarne_trial")
parser.add_argument('--report_path', type=str, dest='report_path', default="reports")
args = parser.parse_args()


def delete_and_create_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


# Load data
config = ConfigReader(join(args.src_path, "config.json"))
gaitup = GaitUp(join(args.src_path, "gaitup"))

# Process individual sets
for counter, sensor_trial in enumerate(config.iterate_over_trials()):
    print(f"Convert set nr: {counter}...")
    azure = AzureKinect(join(args.src_path, "azure", f"{counter + 1:02}_sub", "positions_3d.csv"))
    faros = Faros(join(args.src_path, "faros"), *sensor_trial['faros'])
    gaitup_set = gaitup.cut_data_based_on_index(*sensor_trial['gaitup'])

    # Synchronize signals with respect to Azure Kinect camera
    report_path = join(args.report_path, f"{counter}_azure")
    delete_and_create_directory(report_path)
    gaitup_clock, gaitup_shift = synchronize_signals(azure, gaitup_set, show=True, path=report_path)
    faros_clock, faros_shift = synchronize_signals(azure, faros, show=True, path=report_path)

    global_start = max(azure.timestamps[0], gaitup_clock[0], faros_clock[0])
    global_end = min(azure.timestamps[-1], gaitup_clock[-1], faros_clock[-1])

    print(azure.timestamps)
    print(gaitup_clock)
    print(faros_clock)

    gaitup_set.shift_clock(gaitup_shift)
    faros.add_shift(faros_shift)
    azure.cut_data_based_on_time(global_start, global_end)
    gaitup_set.cut_data_based_on_time(global_start, global_end)
    faros.cut_data_based_on_time(global_start, global_end)
    print(f"Global start: {global_start}, global end: {global_end}")

    plt.plot(azure.timestamps, normalize_signal(azure["pelvis"].to_numpy()[:, 1]))
    plt.plot(faros.timestamps_hr, normalize_signal(faros.hr_data))
    plt.plot(gaitup_set.timestamps, normalize_signal(gaitup_set.get_synchronization_signal()))
    plt.show()