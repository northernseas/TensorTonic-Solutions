import numpy as np

def dbscan(X, eps=0.5, min_samples=5):
    """
    Returns: list of integer labels (-1 for noise)
    """
    X = np.array(X, dtype=np.float64)

    N, D = X.shape

    visited = set()
    cluster = 0
    labels = np.full(N, -1)

    def get_neighbors(x):
        dists = np.sqrt(np.sum((x[None, :] - X) ** 2, axis=-1))
        neighbors = np.where(dists <= eps)[0]
        return neighbors

    for i in range(N):
        if i in visited:
            continue
        visited.add(i)

        neighbors = get_neighbors(X[i])
        if len(neighbors) < min_samples:
            continue

        labels[i] = cluster

        seed_set = set(neighbors) - {i}
        seed_list = list(seed_set)

        j = 0
        while j < len(seed_list):
            q = seed_list[j]
            if q not in visited:
                visited.add(q)
                q_neighbors = get_neighbors(X[q])
                if len(neighbors) >= min_samples:
                    for qn in q_neighbors:
                        if qn not in seed_set:
                            seed_set.add(qn)
                            seed_list.append(qn)
            if labels[q] == -1:
                labels[q] = cluster
            j += 1
        
        cluster += 1

    return labels