import numpy as np

def agglomerative(X, n_clusters=2, linkage='single'):
    """
    Returns: list of integer cluster labels
    """
    X = np.array(X, dtype=np.float64)

    N, D = X.shape

    labels = np.arange(N)

    dists = np.sqrt(
        np.sum(
            (X[:, None, :] - X[None, :, :]) ** 2,
            axis=-1
        )
    ) # (N, N)
    np.fill_diagonal(dists, np.inf)

    while len(np.unique(labels)) > n_clusters:
        clusters = np.unique(labels)
        d_min = float("inf")
        a_cand = None
        b_cand = None

        # Find best pair of clusters
        for a in clusters:
            for b in clusters:
                if a == b:
                    continue

                cluster_dists = dists[labels == a][:, labels == b]
                if linkage == "single":
                    d = np.min(cluster_dists)
                elif linkage == "complete":
                    d = np.max(cluster_dists)
                elif linkage == "average":
                    d = np.mean(cluster_dists)
                else:
                    raise ValueError()

                if d < d_min:
                    d_min = d
                    a_cand = a
                    b_cand = b

        # Merge
        new_id = labels.max() + 1
        labels[labels == a_cand] = new_id
        labels[labels == b_cand] = new_id

    _, labels = np.unique(labels, return_inverse=True)
    
    return labels