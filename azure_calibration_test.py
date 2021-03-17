from src.devices import AzureKinect
from src.rendering import SkeletonViewer

from os.path import join

path = "data/bjarne_trial/"
calibraton_path = join(path, "calibration")
master_azure = join(path, "azure/01_master/positions_3d.csv")
sub_azure = join(path, "azure/01_sub/positions_3d.csv")

azure_master = AzureKinect(master_azure)
azure_master.process_raw_data()

azure_sub = AzureKinect(sub_azure)
azure_sub.process_raw_data()

multi_azure = AzureKinect(master_azure, sub_azure, calibraton_path)

viewer = SkeletonViewer()
viewer.add_skeleton(azure_master.position_data / 1000, azure_master.get_skeleton_connections("src/multicam/multicam.json"))
viewer.add_skeleton(azure_sub.position_data / 1000, azure_master.get_skeleton_connections("src/multicam/multicam.json"))
viewer.add_skeleton(multi_azure.position_data / 1000, azure_master.get_skeleton_connections("src/multicam/multicam.json"))
viewer.show_window()
