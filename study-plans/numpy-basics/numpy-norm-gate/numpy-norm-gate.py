import numpy as np

def norm_gate(X, W, threshold):
    """Returns: np.ndarray of shape (n, k), gated projection where rows below threshold are zeroed"""
    X = np.array(X, dtype=np.float64)
    W = np.array(W, dtype=np.float64)

    Z = X @ W

    Z_l2 = np.sqrt(np.sum(np.power(Z, 2), axis=1)) # (n,)
    # Z_l2 = np.linalg.norm(Z, axis=1)

    Z[Z_l2 < threshold] = 0

    return Z