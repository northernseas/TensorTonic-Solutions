import numpy as np

def softmax_regression(X, y, n_classes, lr=0.01, n_iters=1000):
    """
    Returns: tuple (weights, bias) where weights is a 2D list (d x K) and bias is a list of length K
    """
    X = np.array(X, dtype=np.float64)

    N, D = X.shape
    K = n_classes
    
    # Make y one-hot
    y_ = np.zeros((N, K))
    y_[range(N), y] = 1

    w = np.zeros((D, K))
    b = np.zeros(K)

    for _ in range(n_iters):
        h = X @ w + b  # (N, K)
        z = np.exp(h) / np.sum(np.exp(h), axis=-1, keepdims=True)  # (N, K)

        loss = -np.mean(y_ * np.log(z))

        # dl_dh = (z - y_) / N
        dw = X.T @ (z - y_) / N
        db = np.sum(z - y_, axis=0) / N
        
        w -= lr * dw
        b -= lr * db
    
    return w, b