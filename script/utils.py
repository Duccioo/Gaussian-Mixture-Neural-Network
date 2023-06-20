import matplotlib.pyplot as plt


def plot_AllInOne(
    training_sample, test_sample, pdf_predicted, pdf_true, bins=32, density=True, save=False, show=True, name="figure1"
):
    # Plot delle pdf
    plt.plot(test_sample, pdf_predicted, label="Predicted PDF (GMM)")
    plt.plot(test_sample, pdf_true, label="True PDF (Exponential)")
    plt.hist(training_sample, bins=bins, density=density, alpha=0.5, label="Data")
    plt.title("Gaussian Mixture for Exponential Distribution")
    plt.xlabel("X")
    plt.ylabel("Probability Density")
    plt.legend()

    if save == True:
        plt.savefig(name + ".png")

    if show == True:
        plt.show()


def plot_histo(X):
    plt.hist(
        X,
        bins=32,
        density=True,
        alpha=0.5,
    )
    plt.show()


def subplot():
    pass


def save_plot():
    pass


if __name__ == "__main__":
    pass
