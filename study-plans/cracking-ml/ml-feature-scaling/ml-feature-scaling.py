import numpy as np

def feature_scale(X, method="minmax"):
    """
    Returns: 2D list of scaled values
    """
    X = np.array(X, dtype=np.float64)
    if method == "minmax":
        denom = X.max(axis=0) - X.min(axis=0)
        denom = np.where(denom != 0, denom, 1)
        X = (X - X.min(axis=0)) / denom
    else:
        std = X.std(axis=0)
        std = np.where(std != 0, std, 1)
        X = (X - X.mean(axis=0)) / std
    return X