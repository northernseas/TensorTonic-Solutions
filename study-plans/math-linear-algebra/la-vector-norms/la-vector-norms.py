import numpy as np

def vector_norms(v):
    """
    Returns: float64 array of shape (3,) containing [L1, L2, L-inf] norms.
    """
    v = np.array(v, dtype=np.float64)

    norm_1 = np.linalg.norm(v, ord=1)
    norm_2 = np.linalg.norm(v, ord=2)
    norm_inf = np.linalg.norm(v, ord=np.inf)

    return np.stack((norm_1, norm_2, norm_inf))