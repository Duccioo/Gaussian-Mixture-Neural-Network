import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from sklearn.metrics import r2_score


# Funzione per calcolare la PDF della distribuzione esponenziale bidimensionale con rate 0.5
def exp_pdf(x, y, rate_x, rate_y):
    return rate_x * rate_y * np.exp(-rate_x * x - rate_y * y)


# Generiamo dei dati casuali da una distribuzione esponenziale bidimensionale con rate 0.5
np.random.seed(42)
n_samples = 1000
rate = 0.5
data = np.random.exponential(scale=1 / rate, size=(n_samples, 2))

# Creiamo il modello GMM con 1 componente (perché conosciamo il numero di componenti della distribuzione esponenziale)
gmm = GaussianMixture(n_components=1, covariance_type="full", init_params="kmeans")

# Addestriamo il modello GMM sui dati generati
gmm.fit(data)

# Creiamo un meshgrid per rappresentare la PDF sia vera che approssimata
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)

Z_true = exp_pdf(X, Y, rate_x=0.5, rate_y=0.5)
Z_gmm = np.exp(gmm.score_samples(np.column_stack([X.ravel(), Y.ravel()])))
Z_gmm = Z_gmm.reshape(X.shape)
print("asladad", r2_score(Z_true, Z_gmm))
print("first element", Z_gmm[0][0], Z_true[0][0])

# Grafico 3D per confrontare la PDF vera con quella approssimata dal GMM
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
surf1 = ax.plot_surface(X, Y, Z_true, cmap="viridis", alpha=0.8, label="True PDF")
surf2 = ax.plot_surface(X, Y, Z_gmm, cmap="plasma", alpha=0.8, label="GMM Approximation")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("PDF")
ax.set_title("True PDF vs GMM Approximation")
plt.show()
