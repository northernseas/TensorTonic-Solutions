import numpy as np

def adamw_compare(X, y, lr, beta1, beta2, weight_decay, n_epochs):
    """
    Returns: tuple of (adam_l2_losses, adamw_losses)
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    N, D = X.shape

    w1 = np.zeros(D)
    m1 = np.zeros(D)
    v1 = np.zeros(D)

    w2 = np.zeros(D)
    m2 = np.zeros(D)
    v2 = np.zeros(D)
    
    adam_l2_losses = []
    adamw_losses = []
    
    for t in range(1, n_epochs + 1):
        # Adam + L2 regularization
        loss = np.mean((X @ w1 - y) ** 2)
        adam_l2_losses.append(loss)
        grad = 2 / N * X.T @ (X @ w1 - y) + weight_decay * w1
        m1 = beta1 * m1 + (1 - beta1) * grad
        v1 = beta2 * v1 + (1 - beta2) * (grad ** 2)
        m1_hat = m1 / (1 - beta1 ** t)
        v1_hat = v1 / (1 - beta2 ** t)
        w1 = w1 - lr / np.sqrt(v1_hat + 1e-8) * m1_hat

        # AdamW
        loss = np.mean((X @ w2 - y) ** 2)
        adamw_losses.append(loss)
        grad = 2 / N * X.T @ (X @ w2 - y)
        m2 = beta1 * m2 + (1 - beta1) * grad
        v2 = beta2 * v2 + (1 - beta2) * (grad ** 2)
        m2_hat = m2 / (1 - beta1 ** t)
        v2_hat = v2 / (1 - beta2 ** t)
        w2 = w2 - lr / np.sqrt(v2_hat + 1e-8) * m2_hat - lr * weight_decay * w2

    return adam_l2_losses, adamw_losses