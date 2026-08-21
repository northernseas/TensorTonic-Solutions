import numpy as np

def lasso_ista(X, y, lam, n_iters):
    """
    Returns: list of weight vectors (lists), length n_iters + 1.
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    N, D = X.shape

    w = np.zeros(D)
    weights = [w.tolist()]

    L = np.max(np.linalg.eigh(X.T @ X / N).eigenvalues)
    t = 1 / L
    
    st = lambda x, t: np.sign(x) * np.maximum(np.abs(x) - t, 0) 
    
    for _ in range(n_iters):
        w = st(w - t * (X.T @ (X @ w - y)) / N, lam * t)
        weights.append(w.tolist())
    
    return weights