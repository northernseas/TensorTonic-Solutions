import numpy as np

def gradient_clipping(gradients, max_norm):
    """
    Returns: dict with 'clipped_gradients' (list of lists), 'original_global_norm', 'clipped_global_norm' (floats), 'was_clipped', 'direction_preserved' (bools)
    """
    gradients = [np.array(g, dtype=np.float64) for g in gradients]

    norm = np.sqrt(sum(np.sum(g ** 2) for g in gradients))

    clipped = norm > max_norm
    if clipped:
        gradients_clipped = [g * max_norm / norm for g in gradients]
    else:
        gradients_clipped = [g.copy() for g in gradients]
        
    norm_clipped = np.sqrt(sum(np.sum(g ** 2) for g in gradients_clipped))

    direction_preserved = True
    if clipped:
        a = np.concatenate([g.ravel() for g in gradients])
        b = np.concatenate([g.ravel() for g in gradients_clipped])
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        direction_preserved = abs(sim > 1 - 1e-6)

    return {
        "clipped_gradients": gradients_clipped,
        "original_global_norm": norm,
        "clipped_global_norm": norm_clipped,
        "was_clipped": clipped,
        "direction_preserved": direction_preserved
    }