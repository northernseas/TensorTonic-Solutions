import numpy as np

def mini_batch_training(X, y, weights, biases, lr, epochs, batch_size):
    """
    Returns: list of floats
    """
    X = np.array(X, dtype=np.float64) # (N, D)
    y = np.array(y, dtype=np.float64) # (N, 1)

    N, D = X.shape
    L = len(weights)
    B = batch_size

    weights = [np.array(w, dtype=np.float64) for w in weights]
    biases = [np.array(b, dtype=np.float64) for b in biases]

    losses = []
    
    for _ in range(epochs):
        for batch_start in range(0, N, B):
            batch = np.arange(batch_start, min(N, batch_start + B))
            Xb, yb = X[batch], y[batch]

            # Forward
            z_all = []
            a_all = [Xb]
            for l in range(L):
                z = a_all[-1] @ weights[l].T + biases[l]
                z_all.append(z)
                a = z if l == L - 1 else np.maximum(0, z)
                a_all.append(a)

            # Backward
            dw = [np.zeros_like(w) for w in weights]
            db = [np.zeros_like(b) for b in biases]
            
            # loss = 1 / (2 * B) * np.sum((a_all[-1] - yb) ** 2)
            delta = 1 / len(batch) * (a_all[-1] - yb) # (B, 1)
            for l in range(L)[::-1]:
                # delta       (B, out_dim)
                # a_all[l]    (B, in_dim)
                # weights[l]  (out_dim, in_dim)
                dw[l] += delta.T @ a_all[l]
                db[l] += delta.sum(axis=0)
                if l == 0: continue
                delta = delta @ weights[l] * (z_all[l - 1] > 0) # (B, in_dim)

            for l in range(L):
                weights[l] -= lr * dw[l]
                biases[l] -= lr * db[l]
            
        # Compute loss
        a = X
        for l in range(L):
            z = a @ weights[l].T + biases[l]
            a = z if l == L - 1 else np.maximum(0, z)
        loss = 0.5 * np.mean(np.sum((a - y) ** 2, axis=-1))
        losses.append(loss)

    return np.round(losses, 4)