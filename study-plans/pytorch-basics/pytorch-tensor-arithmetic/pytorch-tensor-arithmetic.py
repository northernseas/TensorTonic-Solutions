import torch

def tensor_op(x, y, op):
    """
    Returns: list (result tensor converted via .tolist())
    """
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    if op == "add":
        res = x + y
    elif op == "multiply":
        res = x * y
    elif op == "matmul":
        res = x @ y
    elif op == "power":
        res = x ** y
    elif op == "max":
        res = torch.max(x, y)
    return res