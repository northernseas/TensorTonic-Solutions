import numpy as np

def winsorize(data, lo_q, hi_q):
    """Returns: np.ndarray of shape (3, m, n), stacked clipped values, lo_mask, hi_mask"""
    data = np.array(data, dtype=np.float64)

    lo = np.percentile(data, q=lo_q, axis=0)
    hi = np.percentile(data, q=hi_q, axis=0)

    clipped = np.clip(data, lo, hi)
    
    return np.stack((
        clipped,
        data < lo,
        data > hi
    ))