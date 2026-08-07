import torch

def softmax(logits):
    """
    Returns: tensor of same shape with softmax probabilities (each row sums to 1)
    """
    logits = logits - logits.max(dim=-1, keepdim=True).values
    logits = torch.exp(logits)
    logits = logits / logits.sum(dim=-1, keepdim=True)
    return logits