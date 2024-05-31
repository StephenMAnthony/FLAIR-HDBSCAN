import numpy as np

# weighted_mean = calc_weighted_mean(values, weights)


def calc_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return np.divide(np.sum(np.multiply(values, weights)), np.sum(weights))
