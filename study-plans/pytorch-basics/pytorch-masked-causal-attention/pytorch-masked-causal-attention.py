import torch
import math

def causal_attention(Q, K, V):
    """
    Returns: masked attention output tensor
    """
    N, T_q, D = Q.size()
    _, T_k, _ = K.size()

    mask = torch.triu(
        torch.full((T_q, T_k), float("-inf")),
        diagonal=1
    )
    
    scores = Q @ K.transpose(-2, -1) / math.sqrt(D) # (N, T, T)
    scores += mask
    weights = torch.softmax(scores, dim=-1)
    attn = weights @ V
    return attn