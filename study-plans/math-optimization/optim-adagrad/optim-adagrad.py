import numpy as np

def adagrad(X, y, lr, n_epochs):
    """
    Returns: tuple of (losses, effective_lrs) per epoch
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    N, D = X.shape
    eps = 1e-8
    
    losses = []
    effective_lrs = []

    w = np.zeros(D)
    G = np.zeros(D)
    
    for _ in range(n_epochs):
        loss = np.mean((X @ w - y) ** 2)
        losses.append(loss)

        grad = 2 / N * X.T @ (X @ w - y)

        G = G + grad ** 2
        w = w - lr / np.sqrt(G + eps) * grad

        effective_lrs.append(lr / np.sqrt(G + eps))

    return losses, effective_lrs