from os.path import join

import numpy as np
import os


def read_calibration_folder(folder_path):
    if not os.path.exists(folder_path):
        raise Exception(f"Path does not exist: {folder_path}")

    main_files = read_file_names('main', folder_path)
    sub_files = read_file_names('sub', folder_path)

    main_points, sub_points = [], []
    for index, main_file in main_files.items():
        if index not in sub_files:
            print(f"Missing calibration file: {main_files[index]}")
            continue

        main_points.extend(read_calibration_file(main_file))
        sub_points.extend(read_calibration_file(sub_files[index]))

    return np.array(main_points), np.array(sub_points)


def read_file_names(prefix, folder_path):
    files = filter(lambda x: x.startswith(prefix) and x.endswith('.txt'), os.listdir(folder_path))
    files = map(lambda x: (int(x[-9:-4]), join(folder_path, x)), files)
    return dict(files)


def read_calibration_file(file_name):
    with open(file_name) as file:
        lines = file.readlines()

    points = list(map(lambda x: list(map(float, x.strip().split(' '))), lines))
    return points


if __name__ == '__main__':
    a, b = read_calibration_folder("../../../../data")
    print(a, b)
