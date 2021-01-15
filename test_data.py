from src.camera import AzureKinect
from src.rendering import SkeletonViewer

import pandas as pd
import numpy as np

azure = AzureKinect("data/positions_3d.csv")
azure.process_raw_data()

data = azure.get_data()
print(data.shape)

viewer = SkeletonViewer()
viewer.add_skeleton(data / 1000)
viewer.show_window()

