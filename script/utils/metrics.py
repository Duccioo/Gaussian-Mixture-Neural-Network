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
    kl_divergence = entropy(true_pdf, predicted_pdf)
    return np.mean(kl_divergence)


def ise_score(true_pdf, predicted_pdf, bin_width=0.01):
    # Calcola le aree dei rettangoli tra le due distribuzioni
    rectangle_areas = (true_pdf - predicted_pdf) ** 2 * bin_width
    ise = np.sum(rectangle_areas)
    return ise


def calculate_metrics(true_pdf, predicted_pdf, round_num=4):
    metrics = {}
    metrics["r2"] = round(r2_score(true_pdf, predicted_pdf), ndigits=round_num)
    metrics["mse"] = round(
        mean_squared_error(true_pdf, predicted_pdf), ndigits=round_num
    )
    metrics["max_error"] = round(max_error(true_pdf, predicted_pdf), ndigits=round_num)
    metrics["ise"] = round(ise_score(true_pdf, predicted_pdf), ndigits=round_num)
    metrics["kl"] = round(
        kl_divergence_score(true_pdf, predicted_pdf), ndigits=round_num
    )
    metrics["evs"] = round(
        explained_variance_score(true_pdf, predicted_pdf), ndigits=round_num
    )

    return metrics
