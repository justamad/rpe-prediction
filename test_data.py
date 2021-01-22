from src.azure import AzureKinect, synchronize_azures
from src.plot import plot_trajectories_for_all_joints
from src.processing import segment_repetitions


master_azure = AzureKinect("data/bjarne.csv")
master_azure.process_raw_data()

# plot_trajectories_for_all_joints(master_azure, "bjarne.png")
segment_repetitions(master_azure, "")
