import numpy as np

def lr_stability_analysis(curvature, w0, n_iters):
    """
    Returns: dict with 'convergence_bound' (float), 'alpha_values' (list of 3), and three trajectory lists of n_iters+1 floats
    """
    convergence_bound = 2 / curvature

    alpha_1 = 0.5 / curvature
    alpha_2 = 1.9 / curvature
    alpha_3 = 2.1 / curvature
    alpha_values = (alpha_1, alpha_2, alpha_3)

    trajectories = []
    for alpha in alpha_values:
        w = w0
        traj = [w0]
        for _ in range(n_iters):
            w = w - alpha * (curvature * w)
            traj.append(w)
        trajectories.append(traj)

    return {
        "convergence_bound": convergence_bound,
        "alpha_values": alpha_values,
        "convergent_trajectory": trajectories[0],
        "oscillatory_trajectory": trajectories[1],
        "divergent_trajectory": trajectories[2],
    }