import numpy as np
import matplotlib.pyplot as plt
from script.utils import PDF, save_dataset, calculate_pdf


def calculate_pdf_from_file(filename, params, training_size=400):
    # Leggi i campioni dal file
    with open(filename, "r") as file:
        lines = file.readlines()
        samples = [float(line.strip()) for line in lines]

    # Crea un array NumPy
    test_X = np.array(samples[0:training_size])
    test_X = test_X.reshape((test_X.shape[0], 1))
    print(test_X.shape)
    fake_Y = np.zeros((len(test_X), len(params)))

    for d, params_dim in enumerate(params):
        for pdf_info in params_dim:
            pdf_type = pdf_info["type"]
            weight = pdf_info["weight"]

            _, fake_Y1 = calculate_pdf(pdf_type, pdf_info, weight, X_input=test_X[:, d])
            fake_Y[:, d] += fake_Y1

    test_Y = fake_Y[:, 0]
    for d in range(1, len(params)):
        test_Y *= fake_Y[:, d]

    test_Y = test_Y.reshape((test_Y.shape[0], 1))

    save_dataset((test_X, test_Y), "training_prof_" + str(training_size) + ".npz", base_dir="")
    return test_X, test_Y


if __name__ == "__main__":
    # Nome del file di input
    file_name = "data/randomized_dataset[1254].txt"

    # Leggi i campioni dal file
    with open(file_name, "r") as file:
        lines = file.readlines()
        samples = [float(line.strip()) for line in lines]

    # Crea un array NumPy
    samples_array = np.array(samples)
    offset_limit = 0.0
    params = [
        [
            {"type": "exponential", "rate": 1, "weight": 0.2},
            {"type": "logistic", "mean": 4, "scale": 0.8, "weight": 0.25},
            {"type": "logistic", "mean": 5.5, "scale": 0.7, "weight": 0.3},
            {"type": "exponential", "mean": -1, "weight": 0.25, "shift": -10},
        ]
    ]
    pdf = PDF(params)

    # pdf = PDF({"type": "exponential", "mean": -1, "offset": -10})

    pdf.generate_training(
        n_samples=5000,
        seed=42,
    )
    # limit_test = (np.min(pdf.training_X) - offset_limit, np.max(pdf.training_X) + offset_limit)
    limit_test = (0 - offset_limit, 10 + offset_limit)
    pdf.generate_test(range_limit=limit_test, stepper=0.1, save_filename="test_prof.npz", base_dir="")

    trainingX, trainingY = calculate_pdf_from_file(file_name, params, training_size=200)

    # Plot delle pdf
    plt.plot(pdf.test_X, pdf.test_Y, label="True PDF", color="green")

    # Genera l'istogramma
    plt.hist(samples_array, bins=32, density=True, alpha=0.7, color="grey")
    plt.scatter(trainingX, trainingY)
    plt.xlabel("Valore")
    plt.ylabel("Densità")
    plt.title("Istogramma della Distribuzione di Probabilità")
    plt.grid(True)
    plt.show()
