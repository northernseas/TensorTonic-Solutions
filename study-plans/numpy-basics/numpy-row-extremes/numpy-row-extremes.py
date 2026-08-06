import numpy as np

def row_extremes(data):
    """Returns: np.ndarray of shape (4, m), rows are max_val, max_col, min_val, min_col"""
    data = np.array(data, dtype=np.float64)
    max_idx = np.argmax(data, axis=1)
    min_idx = np.argmin(data, axis=1)
    return np.stack((
        data[np.arange(len(data)), max_idx],
        max_idx,
        data[np.arange(len(data)), min_idx],
        min_idx
    ))