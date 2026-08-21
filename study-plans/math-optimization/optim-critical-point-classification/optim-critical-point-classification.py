import numpy as np

def classify_critical_point(H):
    """
    Returns: one of 'local_min', 'local_max', 'saddle', 'degenerate'
    """
    H = np.array(H, dtype=np.float64)
    
    e = np.linalg.eigh(H).eigenvalues

    if np.all(e > 1e-6):
        return "local_min"
    elif np.all(e < -1e-6):
        return "local_max"
    elif np.any(np.abs(e) <= 1e-6):
        return "degenerate"
    return "saddle"