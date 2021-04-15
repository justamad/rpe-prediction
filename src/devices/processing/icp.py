import numpy as np


def find_rigid_transformation_svd(points_a, points_b):
    """
    Calculate rigid transformation between two point sets, using singular value decomposition of covariance matrix
    Reference: http://nghiaho.com/?page_id=671
    :param points_a: Nx3 numpy array with reference point set
    :param points_b: Nx3 numpy array with moving point set
    :return: 3x3 rotation matrix, 3x1 trans vector
    """
    # Convert numpy arrays to matrices
    points_a = np.matrix(points_a.T)
    points_b = np.matrix(points_b.T)

    assert len(points_a) == len(points_b), f"Pointclouds should have same length. Got {len(points_a)}, {len(points_b)}"
    num_rows, num_cols = points_a.shape

    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

    num_rows, num_cols = points_b.shape
    if num_rows != 3:
        raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

    # find mean column wise
    centroid_a = np.mean(points_a, axis=1)
    centroid_b = np.mean(points_b, axis=1)

    # subtract mean
    a_mean = points_a - np.tile(centroid_a, (1, num_cols))
    b_mean = points_b - np.tile(centroid_b, (1, num_cols))

    # dot is matrix multiplication for array
    covariance = a_mean * b_mean.T

    # find rotation
    U, S, Vt = np.linalg.svd(covariance)
    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T * U.T

    t = -R * centroid_a + centroid_b
    return R, t
