import numpy as np
import matplotlib.pyplot as plt


def exponential_pdf(x, lambd):
    return lambd * np.exp(-lambd * x)


def kn_nn_estimation(data, x0, kn):
    n = len(data)
    distances = np.abs(data - x0)
    sorted_distances = np.sort(distances)
    kn_density = (kn / n) / (sorted_distances[kn] * 2)
    return kn_density


# Parametri
lambd = 0.5
num_samples = 1000
data = np.random.exponential(scale=1 / lambd, size=num_samples)
x_values = np.linspace(0, 10, 500)
K1 = 4
kn = lambda n: int(K1 * np.sqrt(n))

pdf_estimates = [kn_nn_estimation(data, x, kn(len(data))) for x in x_values]


plt.figure(figsize=(10, 6))
plt.plot(x_values, pdf_estimates, label="Kn-NN Estimation")
plt.plot(x_values, exponential_pdf(x_values, lambd), label="True Exponential PDF")
plt.xlabel("x")
plt.ylabel("PDF")
plt.title(f"Kn-NN Estimation of Exponential PDF with K1 = {K1} and N = {num_samples}")
plt.legend()
plt.grid()
plt.show()
