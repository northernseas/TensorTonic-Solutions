import numpy as np

def euclidean_distance(x, y):
    """
    Returns: float, the Euclidean distance between x and y.
    """
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    dist = np.sqrt(((x - y) ** 2).sum())
    # dist = np.linalg.norm(x - y)
    return dist.item()