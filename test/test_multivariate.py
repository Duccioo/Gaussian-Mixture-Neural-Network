from script.utils import PDF
import numpy as np
import matplotlib.pyplot as plt


def plot_3d_distribution(X_training, X_test, Y_test):
    # Calcola l'istogramma dei campioni di X_training

    # Crea una figura 3D
    fig = plt.figure(figsize=(12, 6))

    # Crea un subplot 3D per l'istogramma
    ax1 = fig.add_subplot(121, projection="3d")
    hist, xedges, yedges = np.histogram2d(X_training[:, 0], X_training[:, 1], bins=(51, 51))
    # np.min(pdf.training_X) - offset_limit, np.max(pdf.training_X) + offset_limit
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")

    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    dx = dy = 0.5
    dz = hist.ravel()

    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Frequenza")
    ax1.set_title("Istogramma 3D")

    x_min, x_max = X_test[:, 0].min(), X_test[:, 0].max()
    y_min, y_max = X_test[:, 1].min(), X_test[:, 1].max()
    z_min, z_max = 0, max(hist.max(), Y_test.max())

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)

    # Crea un subplot 3D per la PDF di X_test
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(X_test[:, 0], X_test[:, 1], Y_test, cmap="viridis", marker="o")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("PDF")
    ax2.set_title("Scatter Plot della PDF della Distribuzione")

    plt.show()


if __name__ == "__main__":
    offset_limit = 0.5

    pdf = PDF(
        [
            [
                {"type": "exponential", "rate": 1, "weight": 0.2},
                {"type": "logistic", "mean": 4, "scale": 0.8, "weight": 0.25},
                {"type": "logistic", "mean": 5.5, "scale": 0.7, "weight": 0.3},
                {"type": "exponential", "mean": -1, "weight": 0.25, "shift": -10},
            ]
        ]
    )

    # pdf = PDF({"type": "exponential", "mean": -1, "offset": -10})

    pdf.generate_training(
        n_samples=5000,
        seed=42,
    )
    limit_test = (np.min(pdf.training_X) - offset_limit, np.max(pdf.training_X) + offset_limit)
    pdf.generate_test(range_limit=limit_test, stepper=0.1)

    # Plot delle pdf
    plt.plot(pdf.test_X, pdf.test_Y, label="True PDF", color="green")
    plt.hist(pdf.training_X, bins=50, density=True, alpha=0.5, label="Data", color="dimgray")
    plt.title("PDF Multivariate")
    plt.xlabel("X")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.show()

    print(pdf.training_X.shape, pdf.test_X.shape, pdf.test_Y.shape)

    # plot_3d_distribution(pdf.training_X, pdf.test_X, pdf.test_Y)
