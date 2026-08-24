import numpy as np

def impute(X, method="mean"):
    """
    Returns: 2D list with NaN values replaced using the specified method
    """
    X = np.array(X, dtype=np.float64)
    X_ = X.copy()

    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if np.all(mask):
            X_[:, j] = 0
        else:
            if method == "mean":
                X_[mask, j] = np.mean(X[~mask, j])
            elif method == "median":
                X_[mask, j] = np.median(X[~mask, j])

    return np.round(X_, 4)