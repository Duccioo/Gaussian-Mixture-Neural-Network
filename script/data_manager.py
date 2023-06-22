import numpy as np
import os


# ---
from model.gm_model import GaussianMixtureModel
from utils import check_base_dir


def generate_training(n_samples: int = 100, rate: float = 1.0, seed: (int or None) = None):
    if seed is not None:
        random_state = np.random.seed(seed)

    X = np.random.exponential(scale=1 / rate, size=n_samples)
    y = rate * np.exp(X * (-rate))
    return X.reshape(-1, 1), y.reshape(-1, 1)


def generate_test(range_limit: tuple = (0, 100), stepper: float = 0.001, rate: float = 1):
    X = np.arange(range_limit[0], range_limit[1], stepper).reshape(-1, 1)
    y = rate * np.exp(X * (-rate))
    return X.reshape(-1, 1), y.reshape(-1, 1)


def generate_training_MLP_Label(X, n_components=4, seed=None):
    model = GaussianMixtureModel(n_components=n_components, seed=seed)
    Y = []
    for indx, sample in enumerate(X):
        X_1 = np.delete(X, indx).reshape(-1, 1)
        model.fit(X_1)
        Y.append(np.exp(model.score_samples(sample.reshape(-1, 1))))
    Y = np.array(Y).reshape(-1, 1)
    return X, Y


def load_training_MLP_Label(X, load_file=False, n_components=4, seed=None, base_dir=["..", "data"]):
    file_path = check_base_dir(base_dir)
    file_path = os.path.join(file_path, load_file)

    if load_file != False and os.path.isfile(file_path):
        X, y = load_dataset(load_file)
    else:
        X, y = generate_training_MLP_Label(X, n_components=n_components, seed=seed)
        if load_file != False:
            save_dataset([X, y], load_file)
    return X, y


def load_training(load_file=False, n_samples=100, rate=1, base_dir=["..", "data"], seed=42):
    file_path = check_base_dir(base_dir)
    file_path = os.path.join(file_path, load_file)

    if load_file != False and os.path.isfile(file_path):
        X, y = load_dataset(load_file)
    else:
        X, y = generate_training(n_samples=n_samples, rate=rate, seed=seed)
        if load_file != False:
            save_dataset([X, y], load_file)
    return X, y


def load_test(load_file=False, range_limit: tuple = (0, 10), stepper=0.001, rate=1.0, base_dir=["..", "data"]):
    file_path = check_base_dir(base_dir)
    file_path = os.path.join(file_path, load_file)

    if load_file != False and os.path.isfile(file_path):
        X, y = load_dataset(load_file)
    else:
        X, y = generate_test(rate=rate, range_limit=range_limit, stepper=stepper)
        if load_file != False:
            save_dataset([X, y], load_file)
    return X, y


def save_dataset(X, filename="dataset_saved.npy", base_path=["..", "data"]):
    full_path = check_base_dir(base_path)
    full_path = os.path.join(full_path, filename)
    np.save(full_path, X, allow_pickle=True)


def load_dataset(filename="dataset_saved.npy", base_path=["..", "data"]):
    full_path = check_base_dir(base_path)
    full_path = os.path.join(full_path, filename)

    dataset = np.load(full_path, allow_pickle=True)
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
