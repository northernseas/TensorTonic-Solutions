import torch
import torch.nn as nn

def train_with_scheduler(model, dataloader, criterion, optimizer, scheduler, num_epochs):
    """
    Returns: dict with 'losses' (list of per-epoch avg loss) and 'lrs' (list of learning rate per epoch)
    """
    losses = []
    lrs = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_lr = 0
        for x, y in dataloader:
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_lr += scheduler.get_last_lr()[0]
        losses.append(epoch_loss / len(dataloader))
        lrs.append(epoch_lr / len(dataloader))
        scheduler.step()

    return {
        "losses": losses,
        "lrs": lrs
    }