import numpy as np

def summarize(data, axis):
    """Returns: np.ndarray of shape (4, k), rows are mean, std, min, max"""    
    data = np.array(data, dtype=np.float64)
    return np.stack((
        np.mean(data, axis),
        np.std(data, axis),
        np.min(data, axis),
        np.max(data, axis),
    ))