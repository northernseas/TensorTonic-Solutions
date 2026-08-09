import numpy as np

def outer_product(u, v):
    """
    Returns: float64 matrix of shape (m, n), the outer product u v^T.
    """
    u = np.array(u, dtype=np.float64)
    v = np.array(v, dtype=np.float64)

    # outer = np.outer(u, v)
    outer = u[:, None] @ v[None, :]
    return outer