from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    max_error,
    explained_variance_score,
)
from scipy.stats import entropy

import numpy as np

# ----


def kl_divergence_score(true_pdf, predicted_pdf):
    kl_divergence = entropy(true_pdf, qk=predicted_pdf)
    # kl_divergence = np.mean(kl_divergence)
    # print(kl_divergence)
    return min(kl_divergence[0], 100000)

    """Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0."""
    epsilon = 0.00001

    #  normalize predicted_pdf perchè la somma non è 1.0
    predicted_pdf = predicted_pdf / np.sum(predicted_pdf)

    # You may want to instead make copies to avoid changing the np arrays.
    true_pdf = true_pdf + epsilon
    predicted_pdf = predicted_pdf + epsilon

    divergence = np.sum(true_pdf * np.log(true_pdf / predicted_pdf))
    return divergence


def ise_score(true_pdf, predicted_pdf, bin_width=0.01):
    # Calcola le aree dei rettangoli tra le due distribuzioni
    rectangle_areas = (true_pdf - predicted_pdf) ** 2 * bin_width
    ise = np.sum(rectangle_areas)
    return ise


def calculate_metrics(true_pdf, predicted_pdf, round_num=10):
    metrics = {}
    metrics["r2"] = round(r2_score(true_pdf, predicted_pdf), ndigits=round_num)
    metrics["mse"] = round(mean_squared_error(true_pdf, predicted_pdf), ndigits=round_num)
    metrics["max_error"] = round(max_error(true_pdf, predicted_pdf), ndigits=round_num)
    metrics["ise"] = round(ise_score(true_pdf, predicted_pdf), ndigits=round_num)
    metrics["kl"] = round(kl_divergence_score(true_pdf, predicted_pdf), ndigits=round_num)
    metrics["evs"] = round(explained_variance_score(true_pdf, predicted_pdf), ndigits=round_num)

    return metrics
