import matplotlib.pyplot as plt
import datetime
import csv
import os


def write_result(
    experiment_type="model selection",
    experiment_params={"hidden_layers": [64, 32, 16], "activation": "relu"},
    best_params={"hidden_layers": [12, 31, 23], "activation": "relu"},
    r2_score=0.0,
    
    mse_score=0.0,
    max_error_score=0.0,
    evs_score=0.0,
    ise_score=0.0,
    k1_score=0.0,
    model_type="GNN+MLP",
    log_name_file="experiment_log.csv",
):
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = [
        date,
        experiment_type,
        model_type,
        r2_score,
        mse_score,
        max_error_score,
        evs_score,
        ise_score,
        k1_score,
        experiment_params,
        best_params,
    ]

    # check if file exist
    if not os.path.isfile(os.path.join("result", "CSV", log_name_file)):
        with open(os.path.join("result", "CSV", log_name_file), "w", newline="") as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    "DATE",
                    "Experiment Type",
                    "Train Type",
                    "R2 Score",
                    "MSE",
                    "Max Error",
                    "EVS",
                    "ISE",
                    "K1 divergence",
                    "Parameters",
                    "Best Params",
                ]
            )

    with open(os.path.join("result", "CSV", log_name_file), "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(log_entry)


def plot_AllInOne(
    training_sample,
    test_sample,
    pdf_predicted_gmm,
    pdf_true,
    bins=32,
    density=True,
    save=False,
    show=True,
    name="figure1",
    pdf_predicted_mlp=None,
):
    # Plot delle pdf
    plt.plot(test_sample, pdf_predicted_gmm, label="Predicted PDF (GMM)")
    plt.plot(test_sample, pdf_true, label="True PDF (Exponential)")
    if pdf_predicted_mlp is not None:
        plt.plot(test_sample, pdf_predicted_mlp, label="Predicted PDF (MLP)")

    plt.hist(training_sample, bins=bins, density=density, alpha=0.5, label="Data", color="gray")
    plt.title("Gaussian Mixture for Exponential Distribution")
    plt.xlabel("X")
    plt.ylabel("Probability Density")
    plt.legend()

    if save == True:
        plt.savefig(name + ".png")

    if show == True:
        plt.show()


def plot_2pdf(x1, y1, x2, y2):
    plt.plot(x1, y1, label="pdf 1")
    plt.plot(x2, y2, label="pdf 2")
    plt.xlabel("X")
    plt.ylabel("Probability Density")
    plt.legend()
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
