import numpy as np

def hessian_convexity_analysis(X, y):
    """
    Returns: dict with 'hessian' (d x d list of lists), 'eigenvalues' (sorted list), 'is_convex' (bool)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    H = 2.0 * X.T @ X

    eigenvalues = np.linalg.eigvalsh(H)

    is_convex = bool(np.all(eigenvalues >= -1e-10))

    return {
        "hessian": H.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "is_convex": is_convex,
    }