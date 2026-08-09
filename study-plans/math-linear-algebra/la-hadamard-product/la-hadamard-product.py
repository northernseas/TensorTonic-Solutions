import numpy as np

def hadamard_product(A, B):
    """
    Returns: ndarray, the element-wise product A * B.
    """
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    return A * B