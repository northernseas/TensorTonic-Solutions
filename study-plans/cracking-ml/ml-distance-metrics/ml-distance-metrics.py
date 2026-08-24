import math
import numpy as np

def distance_metric(x, y, metric, p=2):
    """
    Compute the distance between vectors x and y using the specified metric.
    Returns: float rounded to 4 decimal places
    """
    x = np.array(x)
    y = np.array(y)
    if metric == "euclidean":
        d = np.sqrt(np.sum((x - y) ** 2))
    elif metric == "manhattan":
        d = np.sum(np.abs(x - y))
    elif metric == "cosine":
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        if norm_x == 0 or norm_y == 0:
            return 0.0
        d = 1 - x @ y / (norm_x * norm_y)
    elif metric == "chebyshev":
        d = np.max(np.abs(x - y))
    elif metric == "minkowski":
        d = (np.sum(np.abs(x - y) ** p)) ** (1 / p)
    return round(d, 4)