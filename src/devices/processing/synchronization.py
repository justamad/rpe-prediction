from .utils import normalize_signal, upsample_data
from scipy import signal
from os.path import join

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def synchronize_signals(kinect_camera, imu_sensor, show=True, path=None):
    """
    Minimum in acceleration signal corresponds to maximum peak in positional data.
    Therefore, we first calculate the 2nd derivative from the Azure Kinect position data to get acceleration
    Then we find and synchronize all minima from the IMU acceleration to the Azure Kinect acceleration
    @param kinect_camera:
    @param imu_sensor:
    @param show:
    @param path:
    @return:
    """
    # Find peaks in IMU and Kinect acceleration data
    kinect_clock, kinect_raw, kinect_processed = kinect_camera.get_synchronization_data()
    imu_clock, imu_raw, imu_processed = imu_sensor.get_synchronization_data()

    # Synchronize data by shifting the IMU clock towards Azure Kinect clock
    kinect_signal_upsampled = upsample_data(kinect_processed,
                                            kinect_camera.sampling_frequency,
                                            imu_sensor.sampling_frequency)

    shift = calculate_correlation(kinect_signal_upsampled, imu_processed, imu_sensor.sampling_frequency)
    clock_diff = kinect_clock[0] - imu_clock[0]
    imu_clock = imu_clock + clock_diff + shift

    # Plot the results
    if show:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10), sharey=True)

        # Plot the Kinect data and its gradients
        ax1.plot(kinect_clock, kinect_raw, label="Positions")
        ax1.plot(kinect_clock, normalize_signal(np.gradient(kinect_raw)), label="Velocity")
        ax1.plot(kinect_clock, normalize_signal(np.gradient(np.gradient(kinect_raw))), label="Acceleration")
        ax1.set_title('Kinect Position Data and Derivatives')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Vertical Axis (normalized)")
        ax1.legend()

        # Plot IMU acceleration with Kinect acceleration
        ax2.plot(kinect_clock, kinect_processed, label=f"{kinect_camera}")
        ax2.plot(imu_clock, imu_processed, label=f"{imu_sensor}")
        ax2.set_title(f"Filtered {imu_sensor} Acceleration vs Kinect Acceleration")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Acceleration (normalized)")
        ax2.legend()

        # Plot Faros Acceleration and Kinect positions
        ax3.plot(kinect_clock, kinect_raw, label=f"{kinect_camera}")
        ax3.plot(imu_clock, imu_raw, label=f"{imu_sensor}")
        ax3.set_title(f"Raw {imu_sensor} Acceleration vs. Kinect Positions")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Vertical Axis (normalized)")
        ax3.legend()
        plt.tight_layout()
        if path is not None:
            plt.savefig(join(path, f"{str(imu_sensor)}.png"))
            plt.close()
            plt.cla()
            plt.clf()
        else:
            plt.show()

    return imu_clock, clock_diff + shift


def align_signals_based_on_peaks(reference_peaks, target_peaks, resolution=10):
    """
    Synchronization method based on peak detection algorithm
    @param reference_peaks: peaks found in the reference signal
    @param target_peaks: peaks found in the target signal
    @param resolution: the resolution with which the target signal is moved over the reference signal
    @return shift in seconds
    """
    assert len(target_peaks) == len(reference_peaks), \
        f"Found a different number of peaks in signals: {len(target_peaks)}, {len(reference_peaks)}."

    sign = 1
    if target_peaks[0] > reference_peaks[0]:
        target_peaks, reference_peaks = reference_peaks, target_peaks
        sign = -sign  # If target sequence is starts later than reference sequence invert the sign

    # Figure out the delta in seconds
    diffs = []
    max_value = int(target_peaks[-1] + 1)
    for count in range(0, max_value * resolution):
        diff = np.sum(np.square(reference_peaks - (target_peaks + count / resolution)))
        diffs.append(diff)

    return sign * (np.argmin(diffs) / resolution)


def calculate_correlation(ref_signal, target_signal, sampling_frequency):
    """
    Calculates cross correlation and returns shift for target signal in seconds based on given sampling frequency
    Method assumes equal sampling frequency of both signals
    @param ref_signal: the reference signal
    @param target_signal: the target signal that will be registered to the reference signal
    @param sampling_frequency: the used sampling frequency to determine shift in seconds
    @return shift in seconds for target signal
    """
    corr = signal.correlate(ref_signal, target_signal)
    shift_in_samples = np.argmax(corr) - len(target_signal) - 1
    return shift_in_samples / sampling_frequency
