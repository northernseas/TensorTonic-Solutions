import numpy as np

def scale_cols(data, weights):
    """Returns: np.ndarray of shape (m, n), each column scaled by corresponding weight"""
    data = np.array(data, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    return data * weights[None, :]