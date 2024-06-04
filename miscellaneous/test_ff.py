import numpy as np
import matplotlib.pyplot as plt


def fourier_series_coefficients(X, N):
    """
    Calcola i coefficienti della serie di Fourier per i primi N componenti.

    Parametri:
    X (array-like): Input dei dati.
    N (int): Numero di componenti della serie di Fourier da calcolare.

    Restituisce:
    coefficients (array): Coefficienti della serie di Fourier.
    """

    X = np.asarray(X)
    n_samples = len(X)
    # frequencies = np.fft.fftfreq(n_samples)
    coefficients = np.fft.fft(X)[:N] / n_samples

    return coefficients


# Definisci la funzione esponenziale da approssimare
def exponential_function(x):
    return np.exp(x)


# Parametri per l'approssimazione con Fourier
num_samples = 1000
x_values = np.linspace(-2 * np.pi, 2 * np.pi, num_samples)
num_components = 10

# Calcola i coefficienti della serie di Fourier
coefficients = fourier_series_coefficients(exponential_function(x_values), num_components)


# Calcola l'approssimazione della funzione con la serie di Fourier
def fourier_approximation(x, coefficients):
    result = np.zeros_like(x, dtype=np.complex128)
    for n, c in enumerate(coefficients):
        result += c * np.exp(1j * n * x, dtype=np.complex128)
    return result


# Calcola l'approssimazione con la serie di Fourier
approximated_values = fourier_approximation(x_values, coefficients)

# Plotta la funzione originale e l'approssimazione
plt.plot(x_values, exponential_function(x_values), label="Funzione Esponenziale Originale")
plt.plot(x_values, approximated_values.real, label="Approssimazione Fourier")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Approssimazione di una Funzione Esponenziale con la Serie di Fourier")
plt.legend()
plt.show()
