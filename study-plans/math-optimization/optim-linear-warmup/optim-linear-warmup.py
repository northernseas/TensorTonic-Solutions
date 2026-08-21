import numpy as np

def linear_warmup(X, y, base_lr, warmup_epochs, total_epochs):
    """Train with linear warmup learning rate schedule.

    Returns: see problem description for expected output format
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    N, D = X.shape

    w = np.zeros(D)

    lr_schedules = []
    losses = []

    for epoch in range(total_epochs):
        loss = np.mean((X @ w - y) ** 2)
        losses.append(loss)
        grad = 2 / N * X.T @ (X @ w - y)
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
        else:
            lr = base_lr
        w = w - lr * grad
        lr_schedules.append(lr)

    return lr_schedules, losses