import numpy as np
import matplotlib.pyplot as plt

from utils.data_manager import PDF


def exponential_pdf(x, rate):
    return np.exp(-rate * x)


def kn_nn_estimation(data, x0, kn):
    n = len(data)
    data = data.ravel()
    distances = np.abs(data - x0)
    sorted_distances = np.sort(distances)
    kn_density = (kn / n) / (sorted_distances[kn] * 2)
    return kn_density


if __name__ == "__main__":
    pdf = PDF({"type": "exponential", "mean": 1.0})
    stepper_x_test = 0.01
    n_samples = 100
    seed = 42
    data = np.random.exponential(scale=1.0, size=n_samples)
    # np.random.seed(seed)
    x_training, y_training = pdf.generate_training(n_samples=n_samples, seed=seed)

    x_test, y_test = pdf.generate_test(stepper=stepper_x_test)

    K1 = 1
    print(len(x_training), x_training.shape, "shape of data ", data.shape)
    kn = int(K1 * np.sqrt(len(x_training)))  # Definizione di kn in funzione di n
    print("kn::", kn)

    # x = np.linspace(0, 5, 1000)

    pdf_estimated = [kn_nn_estimation(x_training, point, kn) for point in x_test]

    # pdf_true = np.exp(-x)
    pdf_true = y_test
    plt.plot(x_test, pdf_estimated, label="Estimated PDF")

    plt.plot(x_test, pdf_true, label="True PDF")
    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.title("Kn NN method")
    plt.legend()
    plt.show()
