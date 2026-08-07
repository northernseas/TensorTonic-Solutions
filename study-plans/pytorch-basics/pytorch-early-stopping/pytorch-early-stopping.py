import torch
import torch.nn as nn

def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, max_epochs, patience):
    """
    Returns: dict with 'train_losses' (list), 'val_losses' (list), 'stopped_epoch' (int, 1-indexed)
    """
    best_val_loss = float("inf")
    last_best_epoch = 0
    train_losses = []
    val_losses = []

    epoch = 0
    while epoch < max_epochs:
        epoch += 1

        # Training
        model.train()
        train_loss = 0
        for x, y in train_loader:
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                loss = criterion(model(x), y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            last_best_epoch = epoch
        if last_best_epoch + patience <= epoch:
            break

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "stopped_epoch": epoch
    }