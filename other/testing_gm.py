import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Generazione dei dati di una distribuzione esponenziale
random_state = np.random.seed(33)
n_samples = 100
rate = 1  # tasso di decay dell'esponenziale
X = np.random.exponential(scale=1 / rate, size=n_samples)


# Fit del modello di Gaussian Mixture sui dati
n_components = 8  # Numero di componenti nel modello di Gaussian Mixture
gmm = GaussianMixture(
    n_components=n_components,
    init_params="random",
    covariance_type="full",
    max_iter=1000,
    n_init=10,
    random_state=random_state,
)
gmm.fit(X.reshape(-1, 1))

# Generazione di punti per la predizione della pdf
x_values = np.arange(np.min(X), np.max(X), 0.001).reshape(-1, 1)

# Calcolo della pdf predetta dal modello di Gaussian Mixture
pdf_values = np.exp(gmm.score_samples(x_values))


# Calcolo della vera pdf della distribuzione esponenziale
true_pdf_values = rate*np.exp(x_values * (-rate))

# Plot delle pdf
plt.plot(x_values, pdf_values, label="Predicted PDF (GMM)")
plt.plot(x_values, true_pdf_values, label="True PDF (Exponential)")
plt.hist(X, bins=32, density=True, alpha=0.5, label="Data")
plt.title("Gaussian Mixture for Exponential Distribution")
plt.xlabel("X")
plt.ylabel("Probability Density")
plt.legend()
plt.show()
