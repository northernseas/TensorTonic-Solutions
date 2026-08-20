import numpy as np

def gaussian_log_mean_gradient(x, mu, Sigma):
    """
    Returns: list of k floats, the gradient w.r.t. the mean, each rounded to 6 decimals
    """
    x_ = np.asarray(x, dtype=np.float64)
    mu_ = np.asarray(mu, dtype=np.float64)
    S_ = np.asarray(Sigma, dtype=np.float64)

    # g = np.linalg.inv(S_) @ (x_ - mu_)
    g = np.linalg.solve(S_, x_ - mu_)

    return [round(g_, 6) for g_ in g]