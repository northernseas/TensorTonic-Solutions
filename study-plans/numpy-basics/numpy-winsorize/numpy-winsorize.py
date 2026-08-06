import numpy as np

def winsorize(data, lo_q, hi_q):
    """Returns: np.ndarray of shape (3, m, n), stacked clipped values, lo_mask, hi_mask"""
    data = np.array(data, dtype=np.float64)
    lo_q = np.percentile(data, q=lo_q, axis=0)
    hi_q = np.percentile(data, q=hi_q, axis=0)

    data_new = np.clip(data, lo_q, hi_q)

    lower = np.where(data < lo_q, 1, 0)
    higher = np.where(data > hi_q, 1, 0)
    
    return np.stack((
        data_new,
        lower,
        higher
    ))