from src.devices import AzureKinect

import pandas as pd
import numpy as np


def calculate_feature(df, time):
    pass


def sliding_window(df, window_size, overlap):
    length = len(df) - window_size + 1
    for window in range(length):
        data = df.loc[window:window + window_size - 1, :].copy()
        calculate_feature(data, 1)


azure = AzureKinect("data/arne_flywheel/azure/01_sub/")
sliding_window(azure.position_data, 9, 1)
