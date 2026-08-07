import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    x = torch.tensor(x, dtype=torch.float32)
    if method == "relu":
        res = torch.max(torch.tensor(0), x)
    elif method == "sigmoid":
        res = 1 / (1 + torch.exp(-x))
    elif method == "tanh":
        res = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
        # res = torch.tanh(x)
    elif method == "leaky_relu":
        res = torch.where(x > 0, x, 0.01 * x)
    return res.tolist()