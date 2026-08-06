import numpy as np

def filter_and_extract(data, row_start, row_stop, threshold):
    """
    Returns: 1D ndarray of float64
    """
    arr = np.array(data, dtype=np.float64)[row_start:row_stop]
    return arr[arr > threshold]