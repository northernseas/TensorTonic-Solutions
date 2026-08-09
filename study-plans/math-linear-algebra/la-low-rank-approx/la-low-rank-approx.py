import numpy as np

def low_rank_approximation(A, r):
    """
    Returns: float64 ndarray of shape (m, n), the best rank-r approximation of A.
    """
    A = np.array(A, dtype=np.float64)

    U, s, Vt = np.linalg.svd(A, full_matrices=False, compute_uv=True)

    U = U[:, :r]
    s = s[:r]
    Vt = Vt[:r, :]

    A_r = U @ np.eye(len(s)) * s @ Vt
    return A_r