import numpy as np

class Dual:

    def __init__(self, value, derivative):
        self.v = value
        self.d = derivative

    def __add__(self, other):
        if isinstance(other, Dual):
            return Dual(self.v + other.v, self.d + other.d)
        return Dual(self.v + other, self.d)

    def __radd__(self, other):
        return Dual(self.v + other, self.d)

    def __mul__(self, other):
        if isinstance(other, Dual):
            return Dual(
                self.v * other.v,
                self.v * other.d + self.d * other.v
            )
        return Dual(self.v * other, self.d * other)

    def __rmul__(self, other):
        return Dual(self.v * other, self.d * other)

def dual_sin(d):
    return Dual(np.sin(d.v), d.d * np.cos(d.v))

def dual_exp(d):
    return Dual(np.exp(d.v), d.d * np.exp(d.v))

def dual_number_autodiff(x_val):
    """
    Returns: dict with 'f_val', 'f_deriv_dual', 'f_deriv_analytical' (floats) and 'match' (bool)
    """
    x = Dual(x_val, 1)

    f_x = x * x * dual_sin(x) + dual_exp(x)

    f_val = f_x.v
    f_deriv_dual = f_x.d

    f_deriv_analytical = (
        2 * x_val * np.sin(x_val) +
        np.cos(x_val) * x_val**2 +
        np.exp(x_val)
    )
   
    match = np.abs(f_deriv_dual - f_deriv_analytical) < 1e-10
    
    return {
        "f_val": f_val,
        "f_deriv_dual": f_deriv_dual,
        "f_deriv_analytical": f_deriv_analytical,
        "match": match
    }