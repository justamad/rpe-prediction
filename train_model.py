from rpe_prediction.models import build_multi_branch_model
from rpe_prediction.generator import Generator
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os

n_features = 60
n_frames = 104
n_classes = 5
n_steps = 3
batch_size = 16

exp_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
output_path = os.path.join("reports", exp_time)
os.makedirs(output_path)

model = build_multi_branch_model(n_frames, n_features)
gen = Generator("data/intermediate/", n_steps=n_steps, batch_size=batch_size, window_size=n_frames)
