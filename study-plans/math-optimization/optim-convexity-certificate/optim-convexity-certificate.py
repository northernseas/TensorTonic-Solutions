import numpy as np

def convexity_certificate(H):
    """
    Returns: dict with 'is_convex' (bool) and 'min_eigenvalue' (float, rounded to 6 decimals)
    """
    eigenvalues = np.linalg.eigh(H).eigenvalues
    min_e = np.min(eigenvalues)
    return {
        "is_convex": min_e >= -1e-6,
        "min_eigenvalue": round(min_e, 6),
    }