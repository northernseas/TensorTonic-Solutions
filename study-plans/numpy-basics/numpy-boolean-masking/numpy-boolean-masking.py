import numpy as np

def row_summary(data, threshold):
    """Returns: np.ndarray of shape (3, m, n), stacked element mask, any-filtered, all-filtered"""
    arr = np.array(data, dtype=np.float64)
    res1 = arr > threshold
    res2 = np.where(np.any(arr > threshold, axis=1).reshape(-1, 1), arr, 0)
    res3 = np.where(np.all(arr > threshold, axis=1).reshape(-1, 1), arr, 0)
    return np.stack((res1, res2, res3), axis=0)