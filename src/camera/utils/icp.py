import numpy as np
import matplotlib.pyplot as plt
import logging

from mpl_toolkits.mplot3d import Axes3D


def find_rigid_transformation_svd(
        points_a: np.ndarray,
        points_b: np.ndarray,
        weights: np.ndarray = None,
        show=False,
        path=None
):
    """
    Calculate rigid transformation between two point sets, using singular value decomposition of covariance matrix
    Reference: http://nghiaho.com/?page_id=671
    @param points_a: Nx3 numpy array with reference point set
    @param points_b: Nx3 numpy array with moving point set
    @param weights: A 1D-numpy array that holds weights for all points
    @param show: flag if result should be visualized
    @param path: path to save the image
    @return: 3x3 rotation matrix, 3x1 trans vector
    """
    # Convert numpy arrays to matrices
    matrix_a = np.matrix(points_a.T)
    matrix_b = np.matrix(points_b.T)

    assert len(matrix_a) == len(matrix_b), f"Matrices should have same length. Got {len(matrix_a)}, {len(matrix_b)}"
    num_rows, num_cols = matrix_a.shape

    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = matrix_b.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    if weights is None:
        weights = np.ones((len(points_a), 1))

    centroid_a = (np.sum(weights * points_a, axis=0, keepdims=True) / np.sum(weights)).T
    centroid_b = (np.sum(weights * points_b, axis=0, keepdims=True) / np.sum(weights)).T

    # Subtract centroids from point clouds
    a_mean = matrix_a - np.tile(centroid_a, (1, num_cols))
    b_mean = matrix_b - np.tile(centroid_b, (1, num_cols))

    # Calculate covariance matrix
    W = np.diag(weights.flatten())
    covariance = a_mean * W * b_mean.T

    # Find the rotation using SVD
    U, S, Vt = np.linalg.svd(covariance)
    D = np.diag([1.0, 1.0, np.linalg.det(Vt.T * U.T)])
    R = Vt.T * D * U.T

    # Special reflection case
    if np.linalg.det(R) < 0:
        logging.info("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T * U.T

    t = centroid_b - R * centroid_a

    if show:
        # Check the RMSE of the transformation
        transformed = (R @ matrix_a + t).T
        rmse = np.sqrt(np.sum(np.square(points_b - transformed), axis=1))
        rmse_s = f'RMSE transformation: mean={np.mean(rmse):.2f} mm, std={np.std(rmse):.2f}'

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_a[:, 0], points_a[:, 1], points_a[:, 2], label="Points A")
        ax.scatter(points_b[:, 0], points_b[:, 1], points_b[:, 2], label="Points B")
        ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], label="Transf Points A")
        plt.title(rmse_s)
        plt.legend()
        if path is None:
            plt.show()

        # Clean up resources
        plt.close()
        plt.clf()
        plt.cla()

    return R, t
