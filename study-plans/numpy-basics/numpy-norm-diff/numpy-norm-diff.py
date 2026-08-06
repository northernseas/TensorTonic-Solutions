import numpy as np

def norm_diff(a, b, lo, hi):
    """Returns: np.ndarray of absolute differences after clipping and rescaling to [0, 1]"""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    a = (np.clip(a, lo, hi) - lo) / (hi - lo)
    b = (np.clip(b, lo, hi) - lo) / (hi - lo)

    return np.abs(a - b)