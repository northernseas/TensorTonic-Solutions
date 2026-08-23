import numpy as np

def gmm(X, k, max_iters=100, seed=42):
    """
    Returns: tuple of (labels as list, means as list of lists, weights as list)
    """
    X = np.array(X, dtype=np.float64)

    N, D = X.shape
    K = k

    rng = np.random.RandomState(seed)
    
    weights = np.full(K, 1 / K) # (K)
    means = X[rng.choice(N, size=k, replace=False)].copy() # (K, D)
    covs = np.full((K, D), 1) # (K, D)

    def gaussian_pdf(X, mu, var):
        log_pdf = - 0.5 * np.sum(
            np.log(2 * np.pi * var) +
            (X - mu) ** 2 / var,
            axis=-1
        )
        return np.exp(log_pdf)
    
    for _ in range(max_iters):
        # E-step: compute responsibilities
        R = np.zeros((N, K))
        for j in range(K):
            R[:, j] = weights[j] * gaussian_pdf(X, means[j], covs[j])
        row_sums = R.sum(axis=-1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        R /= row_sums

        # M-step: update weights, means, covariances
        N_j = np.sum(R, axis=0)
        N_j_safe = np.where(N_j > 1e-10, N_j, 1)
        
        weights = N_j / N

        means = (
            np.sum(
                R[:, :, None] *              # (N, K, 1)
                X[:, None, :],               # (N, 1, D)
                axis=0                       # (N, K, D)
            )                                #    (K, D)
            / N_j_safe[:, None]              #    (K, 1)
        )                                    #    (K, D)

        covs = (
            np.sum(
                R[:, :, None] *                             # (N, K, 1)
                (X[:, None, :] - means[None, :, :]) ** 2,   # (N, K, D)
                axis=0                                      # (N, K, D)
            )                                               #    (K, D)
            / N_j_safe[:, None]                             #    (K, 1)
            + 1e-6
        )                                                   #    (K, D)
        covs[N_j <= 1e-10] = 1

    labels = np.argmax(R, axis=-1)
    
    return labels, np.round(means, 4), np.round(weights, 4)