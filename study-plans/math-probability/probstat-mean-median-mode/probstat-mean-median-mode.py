import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Returns: dict with 'mean', 'median', 'mode' as floats.
    """
    x = np.array(x, dtype=np.float64)

    values, counts = np.unique(x, return_counts=True)
    mode = values[np.argmax(counts)]
    
    return {
        "mean": np.mean(x),
        "median": np.median(x),
        "mode": mode,
    }