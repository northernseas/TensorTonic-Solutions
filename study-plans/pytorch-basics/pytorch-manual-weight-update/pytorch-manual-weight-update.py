import torch
import torch.nn as nn

def manual_train_step(model, X, y, criterion, lr):
    """
    Returns: loss value as a Python float
    """
    loss = criterion(model(X), y)
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad
            param.grad.zero_()
    return loss.item()
