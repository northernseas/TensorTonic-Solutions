import numpy as np

def lu_decomposition(A):
    """
    Returns: tuple (L, U) where A = L @ U.
    """
    A = np.array(A, dtype=np.float64)

    n = A.shape[0]

    L = np.eye(n, dtype=np.float64)
    U = A.copy()

    for k in range(n - 1):
        # use U[k, k] as the pivot
        pivot = U[k, k]

        for i in range(k + 1, n):
            # multiplier needed to eliminate U[i, k]
            factor = U[i, k] / pivot

            # store that multiplier in L
            L[i, k] = factor

            # eliminate row i of U
            U[i, :] = U[i, :] - factor * U[k, :]

    return L, U