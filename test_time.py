from src.camera import AzureKinect
from src.rendering import SkeletonViewer
from src.processing import synchronize_azures

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

master_azure = AzureKinect("data/camera_1.csv")
master_azure.process_raw_data()

subord_azure = AzureKinect("data/camera_0.csv")
subord_azure.process_raw_data()

master_data, subord_data = synchronize_azures(master_azure, subord_azure)

connection = master_azure.get_skeleton_connections("src/camera/azure.json")

viewer = SkeletonViewer()
viewer.add_skeleton(master_data / 1000, connection)
viewer.add_skeleton(subord_data / 1000, connection)
viewer.show_window()
