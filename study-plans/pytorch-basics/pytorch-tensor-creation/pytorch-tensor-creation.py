import torch

def create_tensor(method, shape, value=0.0):
    """
    Returns: list
    """
    if method == "zeros":
        tensor = torch.zeros(shape)
    elif method == "ones":
        tensor = torch.ones(shape)
    elif method == "full":
        tensor = torch.full(shape, value)
    return tensor