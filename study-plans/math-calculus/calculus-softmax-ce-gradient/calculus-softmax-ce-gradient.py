import numpy as np

def softmax_cross_entropy_gradient(logits, labels):
    """
    Returns: dict with 'softmax', 'grad_chain_rule', 'grad_direct' (lists), 'loss' (float), 'match' (bool)
    """
    z = np.array(logits, dtype=np.float64)
    y = np.array(labels, dtype=np.float64)

    # Softmax
    m = z.max()
    p = np.exp(z - m) / np.sum(np.exp(z - m))

    # Loss
    # log_p = z - (m + np.log(np.sum(np.exp(z - m))))
    loss = - (y * np.log(p)).sum()

    # Grad with chain rule
    J = np.diag(p) - np.outer(p, p)
    dL_dp = - y / p
    dL_dz = J.T @ dL_dp
    grad_chain_rule = dL_dz

    # Grad direct
    grad_direct = p - y

    match = np.allclose(grad_chain_rule, grad_direct)
    
    return {
        "softmax": p,
        "loss": loss,
        "grad_chain_rule": grad_chain_rule,
        "grad_direct": grad_direct,
        "match": match,
    }