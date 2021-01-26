from src.faros import read_entire_datasource, read_datasource_by_name
from src.azure import AzureKinect

from scipy.signal import butter, sosfiltfilt
from scipy import signal

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


def filter_data(data):
    sos = butter(4, 0.1, output='sos')
    return sosfiltfilt(sos, data)


def normalize(data):
    return (data - data.mean()) / data.std()


def integrate(acceleration):
    velocity = [0]
    time = 0.01  # assuming 100 Hz
    for acc in acceleration:
        velocity.append(velocity[-1] + acc * time)
    del velocity[0]
    return velocity


edf_file = "data/trial_1.EDF"
position_file = "data/positions_3d.csv"

# Read in Faros ECG data
read_entire_datasource(edf_file)
edf_data = read_datasource_by_name(edf_file, 'Accelerometer_X')
edf_data = filter_data(edf_data)
edf_data = normalize(edf_data)
plt.plot(np.arange(len(edf_data)) / 100, edf_data, label="Acc X (Vertical)")
# plt.plot(integrate(edf_data))
# plt.plot(integrate(integrate(edf_data)))

# Read in and process Azure Kinect data
azure_data = AzureKinect(position_file)
azure_data.process_raw_data()
back = azure_data['spine_navel'].to_numpy()[:, 1]
plt.plot(np.arange(len(back)) / 30 + 10.2, normalize(back), label="Kinect Spine (Y)")
# plt.plot(normalize(np.gradient(back)), label="First Derivative")
# plt.plot(normalize(np.gradient(np.gradient(back))), label="Second Derivative")
# back = np.gradient(back, 1)
# back = normalize(back)
# plt.plot(back)

# Calculate Cross Correlation
corr = signal.correlate(back, edf_data)
corr = normalize(corr)
shift_in_samples = np.argmax(corr) - len(edf_data) - 1
print(shift_in_samples)


# plt.plot(edf_data, label="Faros")
# plt.plot(back, label="Azure Position")
# plt.plot(corr, label="Cross Correlation")
plt.legend()
plt.show()
