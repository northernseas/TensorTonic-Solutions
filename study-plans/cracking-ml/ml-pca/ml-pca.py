import numpy as np

def pca(X, n_components=2):
    """
    Returns: tuple of (transformed_data, explained_variance_ratios)
    """
    X = np.array(X, dtype=np.float64)

    N, D = X.shape

    X_centered = X - X.mean(axis=0)
    C = 1 / (N - 1) * X_centered.T @ X_centered

    eigenvalues, eigenvectors = np.linalg.eigh(C)

    idx = np.argsort(eigenvalues)[::-1][:n_components]
    P = eigenvectors[:, idx]

    Y = X_centered @ P

    explained_variance_ratios = eigenvalues[idx] / np.sum(eigenvalues)
    
    return np.round(Y, 4), np.round(explained_variance_ratios, 4)