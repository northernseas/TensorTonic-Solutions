import numpy as np

def least_squares(A, b):
    """
    Returns: float64 array, the solution minimizing ||A @ x - b||^2.
    """
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    return np.linalg.lstsq(A, b, rcond=None)[0]