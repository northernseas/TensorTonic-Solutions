import numpy as np

def compare_correlations(a, b):
    """Returns: np.ndarray of shape (3, n, n), stacked correlation matrices"""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    combined = np.concatenate((a, b), axis=0)
    return np.stack((
        np.corrcoef(a.T),
        np.corrcoef(b.T),
        np.corrcoef(combined.T),
    ))