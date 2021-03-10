from src.devices.azure import AzureKinect, MultiAzure
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

multi_azure = MultiAzure(master_azure, sub_azure, calibraton_path)

viewer = SkeletonViewer()
viewer.add_skeleton(azure_master.get_data() / 1000, azure_master.get_skeleton_connections("src/azure/azure.json"))
viewer.add_skeleton(azure_sub.get_data() / 1000, azure_master.get_skeleton_connections("src/azure/azure.json"))
viewer.add_skeleton(multi_azure.get_data() / 1000, azure_master.get_skeleton_connections("src/azure/azure.json"))
viewer.show_window()
