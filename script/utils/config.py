BASE_RESULT_DIR = ["..", "..", "result"]
BASE_DATA_DIR = ["..", "..", "data"]


MULTIVARIATE_1254 = [
    [
        {"type": "exponential", "rate": 1, "weight": 0.2},
        {"type": "logistic", "mean": 4, "scale": 0.8, "weight": 0.25},
        {"type": "logistic", "mean": 5.5, "scale": 0.7, "weight": 0.3},
        {"type": "exponential", "mean": -1, "weight": 0.25, "shift": -10},
    ]
]

EXPONENTIAL_06 = [
    [
        {"type": "exponential", "rate": 0.6},
    ]
]
