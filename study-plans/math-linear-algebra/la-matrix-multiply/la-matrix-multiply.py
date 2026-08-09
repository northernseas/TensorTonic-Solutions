import numpy as np

def matrix_multiply(A, B):
    """
    Returns: 2-D float64 array, the matrix product A @ B.
    """
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)

    return A @ B