from src.devices.azure import AzureKinect, MultiAzure
from src.devices.gaitup import GaitUp
from src.devices.faros import Faros
from src.devices.processing import normalize_signal, synchronize_signals, calculate_correlation, fill_missing_data, sample_data_uniformly, apply_butterworth_filter_dataframe, find_closest_timestamp, apply_butterworth_filter
