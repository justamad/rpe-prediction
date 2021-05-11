from src.devices.processing import find_rigid_transformation_svd
from os.path import join

import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, dest='path', default="data/cali")
parser.add_argument('--square_size', type=float, dest='square_size', default=50.125)  # unit is in mm
parser.add_argument('--width', type=int, dest='width', default=8)  # width of checkerboard
parser.add_argument('--height', type=int, dest='height', default=5)  # height of checkerboard
parser.add_argument('--marker_height', type=float, dest='marker_height', default=15)  # height of checkerboard
args = parser.parse_args()


def read_files(path, suffix):
    files = filter(lambda x: x.endswith(".txt"), os.listdir(path))
    files = filter(lambda x: suffix in x, files)

    all_points = []
    for file in sorted(files):
        with open(join(path, file)) as content:
            data = list(map(lambda x: list(map(float, x.strip().split(' '))), content.readlines()))
            all_points.extend(data)

    return np.array(all_points)


def filter_zero(points_a, points_b):
    mask_a = np.sum(points_a == 0, axis=1) > 0
    mask_b = np.sum(points_b == 0, axis=1) > 0
    mask = np.logical_and(np.invert(mask_a), np.invert(mask_b))
    return points_a[mask], points_b[mask]


data_sub = read_files(args.path, "sub")
data_main = read_files(args.path, "main")
data_sub, data_main = filter_zero(data_sub, data_main)

rot, trans = find_rigid_transformation_svd(data_sub, data_main, True)
np.savetxt("rot.np", rot)
np.savetxt("trans.np", trans)
