import numpy as np

def eigendecompose(A):
    """
    Returns: tuple (eigenvalues, eigenvectors), sorted by descending magnitude.
    """
    A = np.array(A, dtype=np.float64)

    eigenvalues, eigenvectors = np.linalg.eig(A)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    # Sort
    idx = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Normalize
    eigenvectors /= np.linalg.norm(eigenvectors, axis=0)
    
    return eigenvalues, eigenvectors