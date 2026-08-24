import numpy as np

def forward_pass(x, weights, biases):
    """
    Returns: Dict with "activations" and "pre_activations", values rounded to 4 decimals.
    """
    x = np.array(x, dtype=np.float64)
    weights = [np.array(w, dtype=np.float64) for w in weights]
    biases = [np.array(b, dtype=np.float64) for b in biases]

    N = len(x)    
    L = len(weights)

    pre_activations = []
    activations = [x]

    for l in range(L):
        z = weights[l] @ activations[-1] + biases[l]
        a = z if (l == L - 1) else np.maximum(0, z)

        pre_activations.append(z)
        activations.append(a)

    return {
        "activations": [np.round(a, 4) for a in activations],
        "pre_activations": [np.round(z, 4) for z in pre_activations],
    }