import torch

def gradient_accumulation(w_init, micro_batches, lr, accum_steps):
    """
    Returns: tuple of (updated_weights_list, last_avg_gradient_list)
    """
    w = torch.tensor(w_init, dtype=torch.float32, requires_grad=True)
    last_avg_grad = None

    for step, (x, t) in enumerate(micro_batches):
        x = torch.tensor(x, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.float32)
        l = (w @ x - t) ** 2
        l.backward()

        if (step + 1) % accum_steps == 0:
            last_avg_grad = w.grad.clone() / accum_steps
            with torch.no_grad():
                w -= lr * last_avg_grad
            w.grad.zero_()

    return (w.detach(), last_avg_grad)