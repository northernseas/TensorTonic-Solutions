import numpy as np
import math

def onecycle_train(X, y, max_lr, div_factor, final_div, pct_start, total_epochs):
    """
    Train with one-cycle learning rate policy.

    Returns: dict with base_losses, onecycle_losses, lr_history
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    initial_lr = max_lr / div_factor
    min_lr = initial_lr / final_div
    E_up = math.floor(total_epochs * pct_start)
    E_down = total_epochs - E_up

    N, D = X.shape

    w = np.zeros(D)

    lr_schedules = []
    losses = []

    for e in range(total_epochs):
        loss = np.mean((X @ w - y) ** 2)
        losses.append(loss)
        grad = 2 / N * X.T @ (X @ w - y)

        if e < E_up:
            lr = initial_lr + (max_lr - initial_lr) * (1 - math.cos(math.pi * e / E_up)) / 2
        else:
            lr = min_lr + (max_lr - min_lr) * (1 + math.cos((math.pi * (e - E_up)) / E_down)) / 2
        
        lr_schedules.append(lr)

        w = w - lr * grad

    return {
        "lr_schedule": lr_schedules,
        "losses": losses,
        "initial_lr": initial_lr,
        "min_lr": min_lr,
    }
