from tsfresh.feature_extraction import MinimalFCParameters
from rpe_prediction.plot import plot_sensor_data_for_axes
from rpe_prediction.devices import AzureKinect

import pandas as pd
import matplotlib.pyplot as plt

settings = MinimalFCParameters()

features = pd.read_csv("../../x.csv", delimiter=';', index_col=False)
features = features.join(pd.read_csv("../../y.csv", delimiter=';', index_col=False))

# Iterate over features
for subject_id in range(max(features['group'])):
    subject_data = features.loc[features['group'] == subject_id]

    for feature in settings.keys():

        feature_data = subject_data.filter(regex="pos__" + feature).copy()
        if feature_data.empty:
            continue

        print(feature_data.shape)
        f = f"{subject_id}-{feature}"
        plot_sensor_data_for_axes(feature_data, f, AzureKinect.get_skeleton_joints(), f)
