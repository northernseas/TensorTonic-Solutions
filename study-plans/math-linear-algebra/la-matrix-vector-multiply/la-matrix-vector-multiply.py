import numpy as np

def matrix_vector_multiply(A, x):
    """
    Returns: 1-D float64 array, the product A @ x.
    """
    A = np.array(A, dtype=np.float64)
    x = np.array(x, dtype=np.float64)

    return A @ x