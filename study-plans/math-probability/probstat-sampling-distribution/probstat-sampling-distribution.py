from scipy import stats

def sampling_distribution(mu, sigma, n, threshold):
    """
    Returns: dict with 'mean', 'std_error', 'tail_probability' as floats.
    """
    sampling_mean = round(mu, 4)
    sampling_std = round(sigma / (n ** 0.5), 4)
    prob_below_threshold = round(stats.norm.cdf(
        threshold,
        sampling_mean,
        sampling_std
    ), 4)

    return {
        "sampling_mean": sampling_mean,
        "sampling_std": sampling_std,
        "prob_below_threshold": prob_below_threshold
    }