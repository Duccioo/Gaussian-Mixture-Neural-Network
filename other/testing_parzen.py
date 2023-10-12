import numpy as np
import matplotlib.pyplot as plt

# Definisci la funzione di densità di probabilità esponenziale
def exponential_pdf(x, rate):
    return np.exp(-rate * x)

# Definizione della finestra di Parzen con finestra gaussiana
def parzen_window(x, data, h):
    
    n = len(data)
    window = 1 / (np.sqrt(2 * np.pi) * h) * np.exp(-0.5 * ((x - data) / h) ** 2)
    return np.sum(window) / (n)
    

# Generazione di dati da una PDF esponenziale
np.random.seed(42)
data = np.random.exponential(scale=1.0, size=1000)
 
# Parametri per la finestra di Parzen
h = 0.2  # Larghezza della finestra

# Generazione di punti su cui stimare la PDF
x = np.linspace(0, 5, 1000)

# Calcolo della stima della PDF utilizzando la finestra di Parzen
pdf_estimated = [parzen_window(point, data, h) for point in x]

# Calcolo della vera PDF esponenziale
pdf_true = np.exp(-x)

# Plot dei risultati
plt.plot(x, pdf_estimated, label='Stima Parzen')
plt.plot(x, pdf_true, label='PDF vera')
plt.xlabel('x')
plt.ylabel('PDF')
plt.title('Stima della PDF utilizzando la finestra di Parzen')
plt.legend()
plt.show()
