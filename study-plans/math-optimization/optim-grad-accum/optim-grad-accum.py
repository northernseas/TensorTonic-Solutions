import numpy as np

def gradient_accumulation(X, y, lr, micro_batch, accum_steps, n_epochs):
    """Returns dict with no_accum_losses, accum_losses, no_accum_weights, accum_weights.

    Returns: see problem description for expected output format
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    N, D = X.shape
    B = micro_batch

    w1 = np.zeros(D)
    w2 = np.zeros(D)

    no_accum_losses = []
    accum_losses = []

    for _ in range(n_epochs):
        n_iters = int(np.ceil(N / B))
        grad_accum = np.zeros(D)
        accum_count = 0
        for it in range(n_iters):
            start, end = it * B, min((it + 1) * B, N)
            X_ = X[start:end]
            y_ = y[start:end]

            # Standard Update (no accumulation)
            grad = 2 / len(X_) * X_.T @ (X_ @ w1 - y_)
            w1 = w1 - lr * grad

            # Accumulated Update
            grad_accum += 2 / len(X_) * X_.T @ (X_ @ w2 - y_)
            accum_count += 1
            if accum_count == accum_steps or (it + 1) == n_iters:
                w2 = w2 - lr * grad_accum / accum_count
                grad_accum = np.zeros(D)
                accum_count = 0

        no_accum_losses.append(np.mean((X @ w1 - y) ** 2))
        accum_losses.append(np.mean((X @ w2 - y) ** 2))

    return {
        "no_accum_losses": no_accum_losses,
        "accum_losses": accum_losses,
        "no_accum_weights": w1,
        "accum_weights": w2,
    }