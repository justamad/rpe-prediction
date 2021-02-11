from src.calibration import read_calibration_folder, find_rigid_transformation_svd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from src.calibration import read_calibration_folder, find_rigid_transformation_svd

# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) != 2:
    print("Please specify the input path.")
    exit(1)
    # sys.argv.append("C:\\Users\\Justin\\Desktop\\1613043487674_test")


def multiply_with_matrix(data, matrix, translation=np.array([0, 0, 0])):
    result = matrix * data.reshape(-1, 3).T + translation.reshape(3, 1)
    return result.T


def to_homogeneous(matrix, vector):
    ret = np.eye(4)
    ret[0:3, 0:3] = matrix
    ret[0:3, 3] = vector.reshape(3)
    return ret


main_points, sub_points = read_calibration_folder(sys.argv[1])
R, t = find_rigid_transformation_svd(sub_points, main_points)
sub_check = multiply_with_matrix(sub_points, R, t)

calibration_error = np.sqrt(np.sum(np.square(main_points - sub_check), axis=1))
print(np.mean(calibration_error))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(main_points[:, 0], main_points[:, 1], main_points[:, 2], label="Main")
ax.scatter(sub_points[:, 0], sub_points[:, 1], sub_points[:, 2], label="Sub")
ax.scatter(sub_check[:, 0], sub_check[:, 1], sub_check[:, 2], label="Check")
plt.title("Calibration error. mean: {:4f} mm, std: {:4f}".format(np.mean(calibration_error),
                                                                 np.std(calibration_error)))
plt.legend()
plt.show()
