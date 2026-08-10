import numpy as np

def percentiles(x, q):
    """
    Returns: numpy array of percentile values.
    """
    x = np.array(x, dtype=np.float64)

    return np.percentile(x, q)