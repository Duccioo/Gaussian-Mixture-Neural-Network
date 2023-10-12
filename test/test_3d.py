import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Crea un array di valori X e Y
X = np.linspace(-10, 10, 100)
Y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(X, Y)

# Calcola i valori Z in base alla funzione Z = 2X + Y
Z = 2*X + Y

# Crea una figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Crea il grafico tridimensionale
surf = ax.plot_surface(X, Y, Z, cmap='viridis')

# Aggiungi una barra dei colori
fig.colorbar(surf)

# Etichette degli assi
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Mostra il grafico
plt.show()
