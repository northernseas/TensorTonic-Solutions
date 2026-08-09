import numpy as np

def svd(A):
    """
    Returns: tuple (U, s, Vt) where A = U @ diag(s) @ Vt.
    """
    # A = U @ sigma @ V.T
    # (M, N) = (M, K) @ (K, K) @ (K, N)

    A = np.array(A, dtype=np.float64)

    U, s, V = np.linalg.svd(A, full_matrices=False, compute_uv=True)
    return U, s, V