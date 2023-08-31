import matplotlib.pyplot as plt
import datetime
import csv
import os
import hashlib

BASE_RESULT_DIR = ["..", "..", "result"]


def write_result(
    id: str or int = "000",
    components: int = 4,
    pdf_type: list = [{"type": "exponential", "mean": 0.6}],
    experiment_params: dict = {"hidden_layers": [64, 32, 16], "activation": "relu"},
    best_params: dict or None = None,
    r2_score: float = 0.0,
    mse_score: float = 0.0,
    max_error_score: float = 0.0,
    evs_score: float = 0.0,
    ise_score: float = 0.0,
    k1_score: float = 0.0,
    model_type: str = "GNN+MLP",
    log_name_file: str = "experiment_log.csv",
    n_samples: int = 100,
    dimension: int = 1,
    base_dir=[BASE_RESULT_DIR, "CSV"],
    **kwargs,
):
    write_csv(
        check_colomn="id",
        id=id,
        components=components,
        model=model_type,
        pdf_type=pdf_type,
        samples=n_samples,
        dimension=dimension,
        r2_score=r2_score,
        MSE_score=mse_score,
        max_error_score=max_error_score,
        EVS_score=evs_score,
        ISE_score=ise_score,
        k1_score=k1_score,
        best_params=best_params,
        experiment_params=experiment_params,
        log_name_file=log_name_file,
        base_dir=base_dir,
        **kwargs,
    )


def plot_AllInOne(
    training_sample,
    test_sample,
    pdf_predicted_gmm=None,
    pdf_true=None,
    bins=32,
    density=True,
    save=False,
    show=True,
    name="figure1",
    title="Gaussian Mixture for Exponential Distribution",
    pdf_predicted_mlp=None,
    base_dir=[BASE_RESULT_DIR, "img"],
):
    # Plot delle pdf
    if pdf_predicted_gmm is not None:
        plt.plot(test_sample, pdf_predicted_gmm, label="Predicted PDF (GMM)", color="blue")
    if pdf_true is not None:
        plt.plot(test_sample, pdf_true, label="True PDF", color="green", linestyle="--")
    if pdf_predicted_mlp is not None:
        plt.plot(
            test_sample,
            pdf_predicted_mlp,
            label="Predicted PDF (MLP)",
            color="red",
        )

    plt.hist(training_sample, bins=bins, density=density, alpha=0.5, label="Data", color="dimgray")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Probability Density")
    plt.legend()

    if save == True:
        extension = ".png"
        img_folder_path = check_base_dir(base_dir)
        img_folder_name = os.path.join(img_folder_path, name)
        # chekc if fil already exists
        if os.path.isfile(img_folder_name + extension):
            counter = 1
            img_tmp = img_folder_name
            while os.path.exists(img_tmp + extension):
                img_tmp = f"{img_folder_name}_{counter}"
                counter += 1

            img_folder_name = img_tmp

        plt.savefig(img_folder_name + extension)

    if show == True:
        plt.show()


def generate_unique_id(params: list = [], lenght: int = 10) -> str:
    input_str = ""

    # Concateniamo le stringhe dei dati di input
    for param in params:
        if type(param) is list:
            param_1 = [str(p) if not callable(p) else p.__name__ for p in param]
        else:
            param_1 = str(param)
        input_str += str(param_1)

    # Calcoliamo il valore hash SHA-256 della stringa dei dati di input
    hash_obj = hashlib.sha256(input_str.encode())
    hex_dig = hash_obj.hexdigest()

    # Restituiamo i primi 8 caratteri del valore hash come ID univoco
    return hex_dig[:lenght]


def plot_2pdf(x1, y1, x2, y2, label1="pdf 1", label2="pdf 2"):
    plt.plot(x1, y1, label=label1)
    plt.plot(x2, y2, label=label2)
    plt.xlabel("X")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.show()


def plot_generatedtarget(x1, y1, x2, y2, label1="pdf 1", label2="pdf 2"):
    plt.scatter(x2, y2, label=label2)
    plt.scatter(x1, y1, label=label1)

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
    # args = [item for sublist in args for item in sublist]
    for idx, path in enumerate(args):
        # check if arguments are a list

        if type(path) is list:
            # path = [item for sublist in path for item in sublist if not isinstance(item, str)]
            for micro_path in path:
                if isinstance(micro_path, list):
                    for micro_micro_path in micro_path:
                        full_path = os.path.join(full_path, micro_micro_path)
                else:
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
    check_colomn=False,
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

    if check_colomn is not False:
        with open(full_path, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if (
                    row[check_colomn.replace("_", " ").title()] == kwargs.get(check_colomn)
                    and kwargs.get(check_colomn) in log_entry
                ):
                    return

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
