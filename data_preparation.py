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
parser.add_argument('--dst_path', type=str, dest='dst_path', default="data/intermediate")
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
    set_counter = f"{counter}_azure"
    azure = AzureKinect(join(args.src_path, "azure", f"{counter + 1:02}_sub", "positions_3d.csv"))
    azure.process_raw_data()
    faros = Faros(join(args.src_path, "faros"), *sensor_trial['faros'])
    gaitup_set = gaitup.cut_data_based_on_index(*sensor_trial['gaitup'])

    # Synchronize signals with respect to Azure Kinect camera
    report_path = join(args.report_path, set_counter)
    delete_and_create_directory(report_path)
    gaitup_clock, gaitup_shift = synchronize_signals(azure, gaitup_set, show=True, path=report_path)
    faros_clock, faros_shift = synchronize_signals(azure, faros, show=True, path=report_path)

    global_start = max(azure.timestamps[0], gaitup_clock[0], faros_clock[0])
    global_end = min(azure.timestamps[-1], gaitup_clock[-1], faros_clock[-1])

    gaitup_set.shift_clock(gaitup_shift)
    faros.add_shift(faros_shift)
    azure.cut_data_based_on_time(global_start, global_end)
    gaitup_set.cut_data_based_on_time(global_start, global_end)
    faros.cut_data_based_on_time(global_start, global_end)
    print(f"Global start: {global_start}, global end: {global_end}")

    # azure.filter_data()
    gaitup_set.filter_data()

    plt.plot(azure.timestamps, normalize_signal(azure["pelvis"].to_numpy()[:, 1]), label="kinect")
    plt.plot(faros.timestamps_hr, normalize_signal(faros.hr_data), label="faros")
    plt.plot(gaitup_set.timestamps, normalize_signal(gaitup_set.get_synchronization_signal()), label="gaitup")
    plt.legend()
    plt.ylabel("Normalized Y-Axes")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

    print(len(azure.timestamps), len(faros.timestamps_hr), len(gaitup_set.timestamps))

    # Save the converted data
    dst_path = join(args.dst_path, set_counter)
