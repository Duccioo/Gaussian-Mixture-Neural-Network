import numpy as np
import matplotlib.pyplot as plt


def exponential_pdf(x, rate):
    return np.exp(-rate * x)


def kn_nn_estimation(data, x0, kn):
    n = len(data)
    distances = np.abs(data - x0)
    sorted_distances = np.sort(distances)
    kn_density = (kn / n) / (sorted_distances[kn] * 2)
    return kn_density


np.random.seed(42)
data = np.random.exponential(scale=1.0, size=1000)

kn = 4
K1 = 4
kn = int(K1 * np.sqrt(len(data)))  # Definizione di kn in funzione di n

x = np.linspace(0, 5, 1000)

pdf_estimated = [kn_nn_estimation(data, point, kn) for point in x]

x = np.linspace(0, 5, 1000)


pdf_true = np.exp(-x)

plt.plot(x, pdf_estimated, label="Estimated PDF")

plt.plot(x, pdf_true, label="True PDF")
plt.xlabel("x")
plt.ylabel("PDF")
plt.title("Kn NN method")
plt.legend()
plt.show()
