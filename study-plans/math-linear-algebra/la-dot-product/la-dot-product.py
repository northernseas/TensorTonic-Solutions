import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    if len(x) != len(y):
        raise ValueError("Length of x and y must match")

    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    dot = x.T @ y
    # dot = np.dot(x, y)
    return dot.item()