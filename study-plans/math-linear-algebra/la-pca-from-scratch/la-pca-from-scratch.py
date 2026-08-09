import numpy as np

def pca_project(X, n_components):
    """
    Returns: ndarray of shape (n_samples, n_components), the projected data.
    """
    X = np.array(X, dtype=np.float64)
    
    N, D = X.shape

    # Compute covariance
    X_norm = X - X.mean(axis=0)
    cov = (X_norm.T @ X_norm) / (N - 1)

    # Find eigenvalues/vectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    # Keep largest n_components
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    V = eigenvectors[:, :n_components]

    return X_norm @ V