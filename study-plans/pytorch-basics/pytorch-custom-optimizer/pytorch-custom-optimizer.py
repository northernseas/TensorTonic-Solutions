import torch
import torch.nn as nn

class CustomSGD(torch.optim.Optimizer):
    """
    Returns: loss or None from step()
    """

    def __init__(self, params, lr=0.01, momentum=0.0):
        defaults = {"lr": lr, "momentum": momentum}
        super().__init__(params, defaults)

        for param_group in self.param_groups:
            param_group["v"] = [
                torch.zeros_like(p)
                for p in param_group["params"]
            ]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enabled_grad():
                loss = closure()

        for param_group in self.param_groups:
            lr = param_group["lr"]
            momentum = param_group["momentum"]

            for i, param in enumerate(param_group["params"]):
                if param.grad is None:
                    continue
                if momentum > 0:
                    v = param_group["v"][i]
                    v.mul_(momentum).add_(param.grad)
                    param -= lr * v
                else:
                    param -= lr * param.grad

        return loss