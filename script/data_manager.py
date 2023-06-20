import numpy as np
from model.gm_model import GaussianMixtureModel_bias
import matplotlib.pyplot as plt


def generate_training(n_samples=100, rate=1):
    random_state = np.random.seed(33)
    X = np.random.exponential(scale=1 / rate, size=n_samples)
    y = rate * np.exp(X * (-rate))
    return X, y


def generate_test(limit_x=100, stepper=0.001, rate=1):
    X = np.arange(0, limit_x, stepper).reshape(-1, 1)
    y = rate * np.exp(X * (-rate))
    return X, y

def generate_training_MLP(X):
    pass

def save_dataset(X, filename="dataset_saved.npy"):
    np.save(filename, X)


def load_dataset(filename="dataset_saved.npy"):
    X = np.load(filename, allow_pickle=True)
    return X


if __name__ == "__main__":
    X, y = generate_training()
    X_test, y_test = generate_test()
    
    plt.plot(X_test,y_test)
    plt.show()
    
    save_dataset(X)
    print(X[0:5])

    Y = load_dataset()
    print(Y[0:5])
