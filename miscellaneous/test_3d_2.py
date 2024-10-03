import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
np.random.seed(42)

# Define the parameters of the bivariate normal distribution
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]  # Covariance matrix

# Create a grid of points
x, y = np.mgrid[-3:3:.1, -3:3:.1]
pos = np.dstack((x, y))

# Create the multivariate normal distribution
rv = multivariate_normal(mean, cov)

# Calculate the probability density function
z = rv.pdf(pos)

# Create the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(x, y, z, cmap='viridis', linewidth=0, antialiased=False)

# Customize the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Probability Density')
ax.set_title('Bivariate Normal Distribution')

# Add a color bar
fig.colorbar(surf, shrink=0.5, aspect=5)

# Adjust the viewing angle
ax.view_init(elev=30, azim=45)

# Save the plot as an image file
plt.savefig('bivariate_normal_distribution_3d.png', dpi=300, bbox_inches='tight')

print("The 3D plot has been saved as 'bivariate_normal_distribution_3d.png'")

# Show the plot (optional, comment out if running on a server without display)
# plt.show()