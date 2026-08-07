import math
import torch

def warmup_cosine_schedule(base_lr, warmup_steps, total_steps):
    """
    Returns: list of learning rates
    """
    warmup = base_lr * (torch.arange(warmup_steps) + 1) / warmup_steps
    cosine = base_lr * 0.5 * (1 + torch.cos(
        math.pi * torch.arange(total_steps - warmup_steps) /
        (total_steps - warmup_steps))
    )
    return torch.cat((warmup, cosine)).tolist()