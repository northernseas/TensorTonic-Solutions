import numpy as np

def row_extremes(data):
    """Returns: np.ndarray of shape (4, m), rows are max_val, max_col, min_val, min_col"""
    data = np.array(data, dtype=np.float64)

    return np.stack((
        np.max(data, axis=1),
        np.argmax(data, axis=1),
        np.min(data, axis=1),
        np.argmin(data, axis=1)
    ))