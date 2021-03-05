from src.faros import Faros
from src.azure import AzureKinect
from src.gaitup import GaitUp
from src.processing import synchronize_signals
from src.config import ConfigReader
from os.path import join

import argparse

# Define Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str, dest='src_path', default="data/bjarne_trial")
args = parser.parse_args()

# Load data
config = ConfigReader(join(args.src_path, "config.json"))
faros = Faros(join(args.src_path, "faros"))
gaitup = GaitUp(join(args.src_path, "gaitup"))

# Process individual sets
for counter, sensor_trial in enumerate(config.iterate_over_trials()):
    faros_set = faros.cut_trial(*sensor_trial['faros'])
    azure = AzureKinect(join(args.src_path, f"0{counter + 1}_master/positions_3d.csv"))
    break
