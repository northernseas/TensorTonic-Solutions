def early_stopping(val_losses, patience):
    """
    Returns: dict with 'best_epoch' (int) and 'stop_epoch' (int).
    """
    nb_epochs = len(val_losses)

    best_epoch = 0
    best_val_loss = float("+inf")
    for epoch in range(nb_epochs):
        if val_losses[epoch] < best_val_loss:
            best_val_loss = val_losses[epoch]
            best_epoch = epoch
        if best_epoch + patience <= epoch:
            break
        
    stop_epoch = min(nb_epochs - 1, best_epoch + patience)

    return {
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
    }