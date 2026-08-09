import numpy as np

def gram_schmidt(vectors):
    """
    Returns: float64 array of shape (k, n), orthonormal basis spanning the input space.
    """
    v = np.array(vectors, dtype=np.float64)

    K, N = v.shape

    q = np.zeros((K, N), dtype=np.float64)

    for i in range(K):
        u = v[i].copy()

        for j in range(i):
            u -= np.dot(v[i], q[j]) * q[j]

        q[i] = u / np.linalg.norm(u)

    return q