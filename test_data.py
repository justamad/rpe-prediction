from src.camera import AzureKinect
from src.rendering import SkeletonViewer

import pandas as pd
import numpy as np

azure = AzureKinect("data/camera_0.csv")
azure.process_raw_data()

connection = azure.get_skeleton_connections("src/camera/azure.json")
data = azure.get_data()
print(data.shape)

viewer = SkeletonViewer()
viewer.add_skeleton(data / 1000, connection)
viewer.show_window()

