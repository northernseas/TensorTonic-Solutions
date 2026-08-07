import torch
import math

def initialize_weights(fan_in, fan_out, method):
    """
    Returns: tensor of shape (fan_out, fan_in) with initialized weights
    """
    res = torch.empty(fan_out, fan_in)
    if method == "xavier_uniform":
        std = math.sqrt(6 / (fan_in + fan_out))
        res.uniform_(-std, std)
    elif method == "xavier_normal":
        std = math.sqrt(2 / (fan_in + fan_out))
        res.normal_(0, std)
    elif method == "he_uniform":
        std = math.sqrt(6 / fan_in)
        res.uniform_(-std, std)
    elif method == "he_normal":
        std = math.sqrt(2 / fan_in)
        res.normal_(0, std)
    else:
        raise ValueError()
    return res