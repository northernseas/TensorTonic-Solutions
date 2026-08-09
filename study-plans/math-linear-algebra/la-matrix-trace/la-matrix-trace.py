import numpy as np

def matrix_trace(A):
    """
    Returns: float, the trace (sum of diagonal elements) of A.
    """
    A = np.array(A, dtype=np.float64)

    if A.shape[0] != A.shape[1]:
        raise ValueError()
    
    N = A.shape[0]

    # return np.trace(A)
    
    return A[range(N), range(N)].sum().item()