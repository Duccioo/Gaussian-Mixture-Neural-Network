import matplotlib.pyplot as plt
from script.utils import PDF


if __name__ == "__main__":

    pdf = PDF(default="EXPONENTIAL_06")

    pdf.generate_training(n_samples=1000)
    pdf.generate_test()

    plt.plot(pdf.test_X, pdf.test_Y, label="True PDF", color="green")

    # Genera l'istogramma
    plt.hist(pdf.training_X, bins=32, density=True, alpha=0.7, color="grey")

    # plt.scatter(pdf.training_X, pdf.training_Y)
    plt.xlabel("Valore")
    plt.ylabel("Densità")
    plt.title("Istogramma della Distribuzione di Probabilità")
    plt.grid(True)
    plt.show()
