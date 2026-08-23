import numpy as np

def kmeans(X, k, max_iters=100, seed=42):
    """
    Returns: tuple of (labels as list[int], centroids as list[list[float]])
    """
    X = np.array(X, dtype=np.float64)

    N, D = X.shape

    rng = np.random.RandomState(seed)

    indices = rng.choice(N, size=k, replace=False)
    centroids = X[indices].copy()
    assignments = None
    
    for _ in range(max_iters):
        # Step 1: Determine assignments
        # (N, 1, D) - (1, K, D) -> (N, K)
        dists = np.sum(
            (X[:, None, :] - centroids[None, :, :]) ** 2,
            axis=-1
        )
        # Improvement 1: ||x-c||^2 = ||x||² + ||c||² - 2xc
        # -> so we don't have to allocate (N, K, D) array
        dists = (
            np.sum(X ** 2, axis=-1, keepdims=True) + # (N, 1)
            np.sum(centroids ** 2, axis=-1, keepdims=True).T - # (1, K)
            2 * X @ centroids.T # (N, K)
        )
        assignments = np.argmin(dists, axis=-1) # (N,)

        # Step 2: Update centroids
        # centroids_new = np.zeros_like(centroids)
        # for j in range(k):
        #     centroid_samples = X[assignments == j]
        #     if len(centroid_samples) > 0:
        #         centroids_new[j] = np.mean(centroid_samples, axis=0)
        #     else:
        #         centroids_new[j] = centroids[j]
        # Improvement 2: vectorized update step
        centroids_new = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=int)
        np.add.at(centroids_new, assignments, X)
        np.add.at(counts, assignments, 1)
        valid = counts > 0
        centroids_new[valid] = centroids_new[valid] / counts[valid, None]
        centroids_new[~valid] = centroids[~valid]

        # Check for convergence
        if np.allclose(centroids, centroids_new):
            break
        centroids = centroids_new
    
    return assignments, np.round(centroids, 4)