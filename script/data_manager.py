import numpy as np
from model.gm_model import GaussianMixtureModel_bias
import os


def generate_training(n_samples=100, rate=1, seed=None):
    if seed is not None:
        random_state = np.random.seed(seed)

    X = np.random.exponential(scale=1 / rate, size=n_samples)
    y = rate * np.exp(X * (-rate))
    return X.reshape(-1, 1), y.reshape(-1, 1)


def generate_test(limit_x=100, stepper=0.001, rate=1):
    X = np.arange(0, limit_x, stepper).reshape(-1, 1)
    y = rate * np.exp(X * (-rate))
    return X.reshape(-1, 1), y.reshape(-1, 1)


def generate_training_MLP(X):
    pass


def load_training(load_file=False, n_sample=100, rate=1):
    if load_file != False and os.path.isfile(load_file):
        X, y = load_dataset(load_file)
    else:
        X, y = generate_training(n_samples=n_sample, rate=rate)
    return X, y


def save_dataset(X, filename="dataset_saved.npy"):
    np.save(filename, X, allow_pickle=True)


def load_dataset(filename="dataset_saved.npy"):
    dataset = np.load(filename, allow_pickle=True)
    X = dataset[:][0]
    y = dataset[:][1]
    return X, y


if __name__ == "__main__":
    X, y = generate_training()
    X_test, y_test = generate_test()

    save_dataset([X, y])
    print("X =", X[0:5])
    print("y =", y[0:5], "\n")

    loaded_X, loaded_y = load_dataset()
    print("loaded X = ", loaded_X[0:5])
    print("loaded y = ", loaded_y[0:5])
