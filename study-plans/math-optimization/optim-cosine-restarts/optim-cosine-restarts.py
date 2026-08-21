import numpy as np
import math

def cosine_restarts(X, y, eta_max, eta_min, T_0, T_mult, total_epochs):
    """
    Train with cosine annealing and warm restarts schedule.

    Returns: dict with sgd_losses, cosine_losses, restart_losses, lr_histories
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    N, D = X.shape

    w = np.zeros(D)

    lr_schedules = []
    losses = []

    t_curr = 0
    t_i = T_0

    for _ in range(total_epochs):
        loss = np.mean((X @ w - y) ** 2)
        losses.append(loss)
        grad = 2 / N * X.T @ (X @ w - y)

        lr = eta_min + (eta_max - eta_min) * (
            1 + math.cos(t_curr / t_i * math.pi)
        ) / 2
        lr_schedules.append(lr)

        w = w - lr * grad

        t_curr += 1
        if t_curr >= t_i:
            t_curr = 0
            t_i = int(t_i * T_mult)

    return lr_schedules, losses