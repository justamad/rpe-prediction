#!/bin/bash

# Function to run the Python script with retries
run_script_with_retry() {
    local file="$1"
    local exp_path="$2"
    local retry_count=0

    while true; do
        # Run the script with the specified file
        python train_dl.py --train True --exp_file "$file" --restore_path "$exp_path"

        # Check the exit code of the previous run
        if [ $? -eq 137 ]; then
            # If the exit code is 137 (SIGKILL), print a message and retry
            echo "Trial $((retry_count + 1)) with $file failed with SIGKILL. Retrying..."
            ((retry_count++))
        else
            # Break the loop if the trial was successful
            break
        fi
    done
}

run_script_with_retry rpe_kinect.yaml Kinect
run_script_with_retry rpe_imu.yaml IMU
run_script_with_retry rpe_both.yaml Fusion
