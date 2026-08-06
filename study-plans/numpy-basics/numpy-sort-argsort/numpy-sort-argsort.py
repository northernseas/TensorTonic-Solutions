import numpy as np

def sort_with_indices(data, axis):
    """Returns: np.ndarray of shape (2, m, n), stacked sorted values and sort indices"""
    data = np.array(data, dtype=np.float64)
    values = np.sort(data, axis)
    idx = np.argsort(data, axis)
    return np.stack((
        values,
        idx
    ))