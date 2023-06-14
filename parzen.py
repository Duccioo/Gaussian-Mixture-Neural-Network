import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, expon

def true_pdf(x, distribution):
    if distribution == "uniform":
        return uniform.pdf(x, loc=-2, scale=4)
    elif distribution == "exponential":
        return expon.pdf(x, loc=0, scale=1)
    elif distribution == "gaussian":
        return norm.pdf(x, loc=0, scale=1)

def parzen_window(x, X_train, h):
    n_samples_train = X_train.shape[0]
    n_features = X_train.shape[1]
    density = np.zeros_like(x)
    
    for i, xi in enumerate(x):
        for j in range(n_samples_train):
            diff = xi - X_train[j]
            density[i] += norm.pdf(diff, loc=0, scale=h)
            
        density[i] /= (n_samples_train * h)
    
    return density

def parzen_neural_network(x, X_train, h, hidden_size):
    n_samples_train = X_train.shape[0]
    n_features = X_train.shape[1]
    density = np.zeros_like(x)
    weights = np.random.randn(hidden_size, n_features)
    
    for i, xi in enumerate(x):
        for j in range(hidden_size):
            diff = xi - weights[j]
            density[i] += norm.pdf(diff, loc=0, scale=h)
        
        density[i] /= (hidden_size * h)
    
    return density

# Parametri
distribution = "uniform"  # Uniforme, esponenziale o gaussiana
h = 0.3  # Dimensione finestra di Parzen
hidden_size = 10  # Dimensione strato nascosto (per PNN)
n_samples_train = 100  # Numero di punti di addestramento

# Generazione dei dati di addestramento
if distribution == "uniform":
    X_train = uniform.rvs(loc=-2, scale=4, size=n_samples_train)
elif distribution == "exponential":
    X_train = expon.rvs(loc=0, scale=1, size=n_samples_train)
elif distribution == "gaussian":
    X_train = norm.rvs(loc=0, scale=1, size=n_samples_train)

# Generazione dei dati di test
x = np.linspace(-6, 6, 1000)

# Calcolo delle PDF reali
true_pdf_values = true_pdf(x, distribution)

# Calcolo delle PDF approssimate con Parzen Window
pw_density = parzen_window(x, X_train, h)

# Calcolo delle PDF approssimate con Parzen Neural Network
pnn_density = parzen_neural_network(x, X_train, h, hidden_size)

# Plot delle PDF reali e approssimate
plt.plot(x, true_pdf_values, label="PDF reale")
plt.plot(x, pw_density, label="Parzen Window")
plt.plot(x, pnn_density, label="Parzen Neural Network")
plt.title("Stima della PDF con Parzen Window e Parzen Neural Network")
plt.xlabel("x")
plt.ylabel("PDF")
plt.legend()
plt.show()
