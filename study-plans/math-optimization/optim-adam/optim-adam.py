import numpy as np

def adam(X, y, lr, beta1, beta2, n_epochs):
    """
    Returns: tuple of (losses, final_weights)
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    N, D = X.shape

    losses = []
    final_weights = []
    
    w = np.zeros(D)
    m = np.zeros(D)
    v = np.zeros(D)

    for t in range(1, n_epochs + 1):
        # Compute loss
        loss = np.mean((X @ w - y) ** 2)
        losses.append(loss)

        # Compute grad
        grad = 2 / N * X.T @ (X @ w - y)

        # Update weight
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        w = w - lr / np.sqrt(v_hat + 1e-8) * m_hat

    return losses, w