import numpy as np

def activation_functions(x, activation):
    """
    Returns: list
    """
    x = float(x)

    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    tanh = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    if activation == "relu":
        value = max(0.0, x)
        deriv = 1.0 if x > 0 else 0.0
    elif activation == "leaky_relu":
        value = x if x > 0 else 0.01 * x
        deriv = 1.0 if x > 0 else 0.01
    elif activation == "sigmoid":
        value = sigmoid(x)
        deriv = sigmoid(x) * (1 - sigmoid(x))
    elif activation == "tanh":
        value = tanh(x)
        deriv = 1 - tanh(x) ** 2
    elif activation == "gelu":
        t = tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))
        value = 0.5 * x * (1 + t)
        deriv = 0.5 + 0.5 * t + 0.5 * x * (1 - t ** 2) * (np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x ** 2))
    elif activation == "swish":
        value = x * sigmoid(x)
        deriv = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    
    return round(value, 4), round(deriv, 4)