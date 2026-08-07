import torch
import math

def scaled_dot_product_attention(Q, K, V):
    """
    Returns: attention output tensor
    """
    sqrt_d_k = math.sqrt(Q.size(-1))
    
    # Q: (N, T, D)    K: (N, T, D) -> (N, D, T)
    # scores: (N, T, T)
    scores = Q @ K.transpose(-2, -1) / sqrt_d_k

    # weights: (N, T, T)
    weights = torch.softmax(scores, dim=-1)

    # V: (N, T, D)
    # out: (N, T, D)
    return weights @ V
