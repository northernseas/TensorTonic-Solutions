import numpy as np

def cosine_similarity(a, b):
    """
    Returns: float in [-1, 1], cosine similarity between a and b.
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    dot = np.dot(a, b) / (a_norm * b_norm)
    return dot.item()