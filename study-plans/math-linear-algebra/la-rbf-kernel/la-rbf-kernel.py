import numpy as np

def rbf_kernel_matrix(X, gamma):
    """
    Returns: ndarray of shape (n, n), the RBF kernel matrix.
    """
    X = np.array(X, dtype=np.float64)

    return np.exp(
        -gamma * (
            (X[:, None] - X[None, :]) ** 2).sum(axis=-1)
    )