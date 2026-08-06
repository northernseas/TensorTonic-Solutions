import numpy as np

def sort_with_indices(data, axis):
    """Returns: np.ndarray of shape (2, m, n), stacked sorted values and sort indices"""
    data = np.array(data, dtype=np.float64)
    idx = np.argsort(data, axis)
    # values = np.sort(data, axis)
    values = np.take_along_axis(data, idx, axis)
    return np.stack((values, idx))