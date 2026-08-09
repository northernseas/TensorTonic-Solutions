import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Returns: ndarray, the attention output softmax(Q @ K.T / sqrt(d_k)) @ V.
    """
    Q = np.array(Q, dtype=np.float64)
    K = np.array(K, dtype=np.float64)
    V = np.array(V, dtype=np.float64)

    # attn = softmax(QK^T/sqrt(dk)) V

    sqrt_d_k = np.sqrt(Q.shape[-1])

    scores = Q @ K.T / sqrt_d_k # (N, N)

    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    # weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)

    # without keepdim: (N, N) / (N)->(1, N)
    # with keepdim:    (N, N) / (1, N)
    
    attn = weights @ V

    return attn