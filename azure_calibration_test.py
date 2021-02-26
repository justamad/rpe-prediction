from src.azure import AzureKinect
from src.rendering import SkeletonViewer

from os.path import join

path = "data/bjarne_trial/"
calibraton_path = join(path, "calibration")

azure_master = AzureKinect(join(path, "azure/01_master/positions_3d.csv"))
azure_master.process_raw_data()

azure_sub = AzureKinect(join(path, "azure/01_sub/positions_3d.csv"))
azure_sub.process_raw_data()

viewer = SkeletonViewer()
viewer.add_skeleton(azure_master.get_data() / 1000, azure_master.get_skeleton_connections("src/azure/azure.json"))
viewer.add_skeleton(azure_sub.get_data() / 1000, azure_master.get_skeleton_connections("src/azure/azure.json"))
viewer.show_window()
