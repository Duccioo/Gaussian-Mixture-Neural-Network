import matplotlib.pyplot as plt
import datetime
import csv
import os


def write_result(
    experiment_type="model selection",
    experiment_params={"hidden_layers": [64, 32, 16], "activation": "relu"},
    best_params={"hidden_layers": [12, 31, 23], "activation": "relu"},
    r2_score=0.0,
    rate=1.0,
    mse_score=0.0,
    max_error_score=0.0,
    evs_score=0.0,
    ise_score=0.0,
    k1_score=0.0,
    model_type="GNN+MLP",
    log_name_file="experiment_log.csv",
    base_dir=["..", "result", "CSV"],
):
    write_csv(
        experiment_type=experiment_type,
        model_type=model_type,
        rate=rate,
        r2_score=r2_score,
        MSE_score=mse_score,
        max_error_score=max_error_score,
        EVS_score=evs_score,
        ISE_score=ise_score,
        k1_score=k1_score,
        experiment_params=experiment_params,
        best_params=best_params,
        log_name_file=log_name_file,
        base_dir=base_dir,
    )


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


def check_base_dir(*args):
    # take the full path of the folder
    absolute_path = os.path.dirname(__file__)

    full_path = absolute_path
    for idx, path in enumerate(args):
        # check if arguments are a list
        if type(path) is list:
            for micro_path in path:
                full_path = os.path.join(full_path, micro_path)

        else:
            full_path = os.path.join(full_path, path)

        # check the path exists
        if not os.path.exists(full_path):
            os.makedirs(full_path)

    return full_path


def write_csv(
    log_name_file="test.csv",
    base_dir="",
    **kwargs,
):
    """
    generate a .csv file to log information from the experiments
    """

    # check the path and create the dir
    base_path = check_base_dir(base_dir)
    # get the All path to the csv file
    full_path = os.path.join(base_path, log_name_file)

    # generate timestamp
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = [date]
    log_title = ["Date"]
    for key, element in kwargs.items():
        log_entry.append(element)
        log_title.append(key.replace("_", " ").title())

    # check if file exist
    if not os.path.isfile(full_path):
        with open(full_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(log_title)

    with open(full_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(log_entry)


def subplot():
    pass


def save_plot():
    pass


if __name__ == "__main__":
    absolute_path = os.path.dirname(__file__)
    relative_path = "prova"
    full_path = os.path.join(absolute_path, relative_path)
    print(absolute_path)
    print(full_path)
    # check_base_dir("prova", "dati", ["prova1"])
    write_result(experiment_type="AAA", experiment_params="No", log_name_file="test.csv")
