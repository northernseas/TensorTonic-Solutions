import numpy as np

def dropout(X, mask, drop_prob, mode):
    """
    Returns: 2D list with values rounded to 4 decimal places.
    """
    X = np.array(X, dtype=np.float64)

    N, D = X.shape
    
    if mode == "train" and drop_prob > 0:
        # mask = np.random.rand(N, D) > drop_prob
        X = X * mask / (1 - drop_prob)

    return np.round(X, 4)