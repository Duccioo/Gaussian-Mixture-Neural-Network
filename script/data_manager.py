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


def generate_training_MLP_Label(X, n_components=4, seed=None):
    model = GaussianMixtureModel_bias(n_components=n_components, seed=seed)
    Y = []
    for indx, sample in enumerate(X):
        X_1 = np.delete(X, indx).reshape(-1, 1)
        model.fit(X_1)
        Y.append(np.exp(model.score_samples(sample.reshape(-1, 1))))
    Y = np.array(Y).reshape(-1, 1)
    return X, Y


def load_training_MLP_Label(X, load_file=False, n_components=4, seed=None):
    if load_file != False and os.path.isfile(load_file):
        X, y = load_dataset(load_file)
    else:
        X, y = generate_training_MLP_Label(X, n_components=n_components, seed=seed)
        if load_file != False:
            save_dataset([X, y], load_file)
    return X, y


def load_training(load_file=False, n_sample=100, rate=1):
    if load_file != False and os.path.isfile(load_file):
        X, y = load_dataset(load_file)
    else:
        X, y = generate_training(n_samples=n_sample, rate=rate)
        if load_file != False:
            save_dataset([X, y], load_file)
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

    print("X =", X[0:5])
    print("y =", y[0:5], "\n")
    save_dataset([X, y])

    loaded_X, loaded_y = load_dataset()
    print("loaded X = ", loaded_X[0:5])
    print("loaded y = ", loaded_y[0:5])

    x1, y1 = load_training_MLP_Label(X, "training_mlp.npy", n_components=4, seed=33)
    print(x1[0:5], y1[0:5])
