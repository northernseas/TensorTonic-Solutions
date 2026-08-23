import numpy as np

def kmeans_plusplus(X, k, seed=42):
    """
    Returns: list of centroids, each a list of floats rounded to 4 decimal places
    """
    X = np.array(X, dtype=np.float64)

    N, D = X.shape
    K = k
    
    centroids = []

    rng = np.random.RandomState(seed)
    
    c = X[rng.choice(N)].copy()
    centroids.append(c)

    for k in range(K - 1):
        C = np.array(centroids)
        dists = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=-1)
        max_dists = np.min(dists, axis=-1) # (N, K) -> (N,)        
        probs = max_dists / np.sum(max_dists)
        c = X[rng.choice(N, p=probs)].copy()
        centroids.append(c)

    return np.round(centroids, 6)