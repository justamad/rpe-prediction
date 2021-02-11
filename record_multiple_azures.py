import subprocess
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--m', type=str, dest='m', default="000080693212")
parser.add_argument('--s', type=str, dest='s', default="000226493212")
args = parser.parse_args()

print(f"Recorder Tool started in folder {os.getcwd()}")

cmd = '"C:\\Program Files\\Azure Kinect SDK v1.4.1\\tools\\k4arecorder.exe"'

# Check that two Azure Kinect cameras are connected
proc = subprocess.Popen(f"{cmd} --list", stdout=subprocess.PIPE)
device_results = proc.communicate()[0].decode("utf-8").strip()
print(device_results)
serials = map(lambda x: x.split("\t")[:2], device_results.split("\n"))
serials = list(map(lambda x: (x[0].replace("Index:", ""), x[1].replace("Serial:", "")), serials))

# Check that both Kinects are connected and associate device
assert len(serials) == 2, "Two Azure Kinect cameras should be connected!"
for cam_id in [args.m, args.s]:
    assert cam_id in [x[1] for x in serials], f"Azure Kinect {cam_id} not found"

master = [x for x, y in serials if y == args.m][0]
subordinate = [x for x, y in serials if y == args.s][0]

print("Master: ", master)
print("Sub: ", subordinate)

# Open Kinect Camera streams
counter = 0

while True:
    print("Press ENTER to start recording")
    a = input()

    cmd_1 = f'{cmd} --device {subordinate} --imu OFF --external-sync Subordinate --sync-delay 1000 sub_{counter}.mkv'
    cmd_2 = f'{cmd} --device {master} --imu OFF --external-sync Master master_{counter}.mkv'

    # Wait for processes to finish
    p_1 = subprocess.Popen(cmd_1, shell=False)
    time.sleep(0.5)
    p_2 = subprocess.Popen(cmd_2, shell=False)

    p_1.wait()
    p_2.wait()
    counter += 1
