import numpy as np

def rmsprop(X, y, lr, decay, n_epochs):
    """
    Returns: tuple of (losses, effective_lrs) per epoch
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    N, D = X.shape

    w = np.zeros(D)
    E = np.zeros(D)
    
    losses = []
    effective_lrs = []

    for _ in range(n_epochs):
        # Compute loss
        loss = np.mean((X @ w - y) ** 2)
        losses.append(loss)

        # Compute grad
        grad = 2 / N * X.T @ (X @ w - y)

        # Update weight
        E = decay * E + (1 - decay) * (grad ** 2)
        w = w - lr / np.sqrt(E + 1e-8) * grad

        # Store effective lr
        effective_lrs.append(lr / np.sqrt(E + 1e-8))
    
    return losses, effective_lrs