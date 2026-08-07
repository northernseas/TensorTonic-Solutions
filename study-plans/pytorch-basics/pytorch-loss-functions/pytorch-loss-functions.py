import torch

def compute_loss(pred, target, method, delta=1.0):
    """
    Returns: float, the mean loss value
    """
    pred = torch.tensor(pred, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)
    if method == "mse":
        loss = (pred - target) ** 2
    elif method == "cross_entropy":
        n = torch.arange(len(target))
        l = pred - torch.logsumexp(pred, dim=-1, keepdim=True)
        loss = -l[n.int(), target.int()]
    elif method == "huber":
        alpha = pred - target
        loss = torch.where(
            torch.abs(alpha) <= delta,
            0.5 * (alpha ** 2),
            delta * (torch.abs(alpha) - 0.5 * delta)
        )
    return loss.mean().item()