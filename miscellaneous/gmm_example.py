import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import seaborn as sns

# Generate sample data
np.random.seed(42)
n_samples = 1000
weights = np.array([0.53, 0.27, 0.08, 0.11])
means = np.array([10.58, 9.65, 11.82, 8.61])
stds = np.array([0.35, 0.36, 0.31, 0.52])

data = np.concatenate(
    [
        np.random.normal(loc=mean, scale=std, size=int(n_samples * weight))
        for mean, std, weight in zip(means, stds, weights)
    ]
)

# Fit Gaussian Mixture Model
n_components = 4
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(data.reshape(-1, 1))

# Generate points for plotting
x = np.linspace(data.min() - 1, data.max() + 1, 1000).reshape(-1, 1)

# Get GMM predictions
y_gmm = np.exp(gmm.score_samples(x))
y_components = np.exp(gmm._estimate_weighted_log_prob(x))

# Plot the results
sns.set_style("darkgrid")
plt.figure(figsize=(12, 8))

# Plot the actual data histogram
plt.hist(data, bins=50, density=True, alpha=0.5, color="gray", label="Actual Data")

# Plot the GMM
plt.plot(x, y_gmm, "r-", linewidth=2, label="Gaussian Mixture Model")

# Plot individual Gaussian components
colors = ["blue", "orange", "green", "purple"]
for i, (y, color) in enumerate(zip(y_components.T, colors)):
    plt.fill_between(
        x.ravel(),
        y,
        alpha=0.3,
        color=color,
        label=f"μ={gmm.means_[i][0]:.2f}, σ={np.sqrt(gmm.covariances_[i][0][0]):.2f}, w={gmm.weights_[i]:.2f}",
    )

plt.xlabel("log(Weights)")
plt.ylabel("Density")
plt.title("Gaussian Mixture Model Estimation")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gmm_pdf_estimation.png", dpi=300)
plt.show()

print("The plot has been saved as 'gmm_pdf_estimation.png'")
