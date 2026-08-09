import numpy as np

def whiten(X):
    """
    Returns: ndarray, the whitened data with identity covariance.
    """
    X = np.array(X, dtype=np.float64)

    N, D = X.shape
    
    X_ = X - X.mean(axis=0)

    C = X_.T @ X / (N - 1)
    
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    D = np.diag(eigenvalues)
    D = np.where(D < 1e-10, 0, 1 / np.sqrt(D))

    return X_ @ eigenvectors @ D