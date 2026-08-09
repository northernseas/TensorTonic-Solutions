import numpy as np

def matrix_determinant(A):
    """
    Returns: float, the determinant of square matrix A.
    """
    def det(A):
        N = A.shape[0]
        if N == 1:
            return A[0, 0]
        res = 0
        for j in range(N):
            M = np.concatenate((A[1:,:j], A[1:,j+1:]), axis=1)
            res += (-1) ** j * A[0, j] * det(M)
        return res
    
    A = np.array(A, dtype=np.float64)
    return np.linalg.det(A)
    # return det(A).item()