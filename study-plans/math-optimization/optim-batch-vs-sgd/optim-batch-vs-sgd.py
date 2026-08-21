import numpy as np

def batch_gd_compare(X, y, batch_sizes, n_epochs, lr, seed):
    """
    Returns: list of loss curves (one list per batch size).
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    N, D = X.shape
    
    all_losses = []
    for bs in batch_sizes:
        losses = []
        w = np.zeros(D)
        rng = np.random.RandomState(seed)
        n_iters = int(np.ceil(len(X) / bs))
        for _ in range(n_epochs):
            perm = rng.permutation(N)
            for it in range(n_iters):
                start = it * bs
                end = min(N, (it + 1) * bs)
                X_batch = X[perm][start:end]
                y_batch = y[perm][start:end]
                bs_actual = end - start
                g = 2 / bs_actual * X_batch.T @ (X_batch @ w - y_batch)
                w = w - lr * g
            loss = np.mean((X @ w - y) ** 2)
            losses.append(loss)
        all_losses.append(losses)
    
    return all_losses