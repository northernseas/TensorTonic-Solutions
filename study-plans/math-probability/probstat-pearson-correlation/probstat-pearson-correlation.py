import numpy as np

def pearson_correlation(X):
    """
    Returns: ndarray, the Pearson correlation matrix.
    """
    X = np.array(X, dtype=np.float64)

    # return np.corrcoef(X.T)
    
    X_ = X - X.mean(axis=0)
    cov = (X_.T @ X_) / (X.shape[0] - 1)

    std = np.sqrt(np.diag(cov)) # (D,)

    corr = cov / (std[:, None] * std[None, :])

    return corr