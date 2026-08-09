import numpy as np

def matrix_transpose(A):
    """
    Returns: ndarray, the transpose of A.
    """
    A = np.array(A, dtype=np.float64)

    A_T = np.empty((A.shape[1], A.shape[0]))

    for i in range(A_T.shape[0]):
        for j in range(A_T.shape[1]):
            A_T[i, j] = A[j, i]

    return A_T