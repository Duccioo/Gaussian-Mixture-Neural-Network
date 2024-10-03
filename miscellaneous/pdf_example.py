import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


def plot_pdf(x, pdf, name, parameters):
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf, "r-", lw=2, label=f"{name} PDF")
    plt.title(f"{name} Probability Density Function\n{parameters}")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{name.lower().replace(" ", "_")}_pdf.png')
    plt.close()


def main() -> None:

    # Normal distribution
    x = np.linspace(-5, 5, 1000)
    mu, sigma = 0, 1
    pdf = stats.norm.pdf(x, mu, sigma)
    plot_pdf(x, pdf, "Normal", f"μ = {mu}, σ = {sigma}")

    # Exponential distribution
    x = np.linspace(0, 5, 1000)
    lambda_param = 1
    pdf = stats.expon.pdf(x, scale=1 / lambda_param)
    plot_pdf(x, pdf, "Exponential", f"λ = {lambda_param}")

    # Gamma distribution
    x = np.linspace(0, 20, 1000)
    k, theta = 2, 2
    pdf = stats.gamma.pdf(x, k, scale=theta)
    plot_pdf(x, pdf, "Gamma", f"k = {k}, θ = {theta}")

    # Beta distribution
    x = np.linspace(0, 1, 1000)
    alpha, beta = 2, 5
    pdf = stats.beta.pdf(x, alpha, beta)
    plot_pdf(x, pdf, "Beta", f"α = {alpha}, β = {beta}")

    # Uniform distribution
    x = np.linspace(-0.5, 1.5, 1000)
    a, b = 0, 1
    pdf = stats.uniform.pdf(x, loc=a, scale=b - a)
    plot_pdf(x, pdf, "Uniform", f"a = {a}, b = {b}")

    print("All PDF plots have been saved as separate PNG files.")


if __name__ == "__main__":
    main()
