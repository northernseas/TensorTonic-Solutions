import numpy as np

def pca_project(X, n_components):
    """Returns: ndarray of shape (n_samples, n_components), the projected data."""
    X = np.array(X, dtype=float)
    mean = X.mean(axis=0)
    X_centered = X - mean
    cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns ascending order, reverse for descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    V_k = eigenvectors[:, :n_components]
    return X_centered @ V_k
