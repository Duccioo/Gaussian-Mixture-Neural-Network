import os
from datetime import datetime

from rich import box
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt

import seaborn as sns

from attrs import define, field

# ---
from .utils import check_base_dir, generate_unique_id, unique_dir, write_csv
from utils.metrics import calculate_metrics
from utils.data_manager import PDF
from .config import BASE_RESULT_DIR


def console_bullet_list(console, list_elem: list = []):
    if isinstance(list_elem, list):
        for element in list_elem:
            console.print("- ", element)


def console_table_list_of_dict(
    console, list_of_dict: list = [{}], inplace: bool = True
):
    table = Table(show_header=True, header_style="bold magenta", box=box.MARKDOWN)
    for key, value in list_of_dict[0].items():
        table.add_column(key)

    for dict in list_of_dict:
        row = []
        for key, value in dict.items():
            row.append(str(value))
        table.add_row(*row)

    if inplace:
        console.print(table)

    return table


def console_table_dict(
    console, dict: dict = {}, header: tuple = ("KEY", "VALUE"), inplace: bool = True
):
    table = Table(show_header=True, header_style="bold magenta", box=box.MARKDOWN)
    table.add_column(header[0])
    table.add_column(header[1])
    for key, value in dict.items():
        table.add_row(key, str(value))

    if inplace:
        console.print(table)
    return table


def console_matrix(console, matrix: list = [], inplace: bool = True):
    table = Table(show_header=True, header_style="bold magenta", box=box.MARKDOWN)
    num_col = len(matrix[0])
    for i in range(num_col):
        table.add_column(str(i + 1))
    for row in matrix:
        row = [str(i) for i in row]
        table.add_row(*row)
    if inplace:
        console.print(table)
    return table


@define(slots=True)
class Summary:

    # Experiment Params
    id_experiment: str = field(init=False)
    date_experiment: str = field(init=False)
    experiment_name: str = field(default="Exanmple 1")

    # -- Path experiement:
    # folder for result
    base_dir: str = field(default=check_base_dir(BASE_RESULT_DIR))

    # folder for that type of experiment
    experiment_dir: str = field(init=False)

    # folder for specific single experiment MARKDOWN
    filename_summary_md_path: str = field(init=False)

    # PDF
    pdf: PDF = field(factory=PDF)

    # Dataset params
    dataset_params: dict = field(factory=dict)
    n_samples: int = field(init=False)
    seed: int = field(init=False)

    # Model params
    model_type: str = field(default="Parzen Windows")
    model_subtype: str = field(default="Parzen Windows Base")
    model_params: dict = field(factory=dict)
    model_metrics: dict = field(factory=dict)

    # Target params
    target_type: str = field(init=False)
    target_params: dict = field(factory=dict)

    # Training params
    train_params: dict = field(factory=dict)

    def __init__(
        self,
        experiment: str = "example 1",
        model_type: str = "Parzen Windows",
        pdf: PDF = None,
        dataset_params: dict = None,
        model_params: dict = None,
        train_params: dict = None,
        target_params: dict = None,
        overwrite: bool = False,
    ):

        self.n_samples = dataset_params["n_samples"]
        self.seed = dataset_params["seed"]
        self.dataset_params = dataset_params

        self.pdf = pdf
        self.model_type = model_type
        self.experiment_name = experiment

        self.model_params = model_params

        if model_type in [
            "Parzen Windows",
            "Parzen Window",
            "GMM",
            "KNN",
            "Parzen",
            "PARZEN",
            "parzen",
        ]:
            self.target_params = None
            self.train_params = None
            self.target_type = None
        else:
            print("Model selected as a MLP")
            self.target_type = dataset_params["target_type"]
            self.target_params = target_params
            self.train_params = train_params

        self.date_experiment = datetime.now().strftime("%Y-%m-%d %H-%M")

        self.id_experiment = str(
            generate_unique_id(
                [
                    self.seed,
                    pdf.params,
                    pdf.default,
                    self.n_samples,
                    dataset_params,
                    model_params,
                    train_params,
                    target_params,
                    model_type,
                ],
                lenght=8,
            )
        )

        self.base_dir = check_base_dir(BASE_RESULT_DIR)
        path_subtype = check_base_dir(BASE_RESULT_DIR, model_type)
        if overwrite:
            self.experiment_dir = check_base_dir(
                path_subtype, self.id_experiment + " " + experiment
            )
        else:
            self.experiment_dir = check_base_dir(
                unique_dir(
                    os.path.join(path_subtype, self.id_experiment + " " + experiment)
                )
            )

        self.filename_summary_md_path = os.path.join(
            self.experiment_dir,
            f"summary_{self.id_experiment}.md",
        )

        with open(self.filename_summary_md_path, "w") as file:
            console = Console(file=file)
            console.print(f"# Experiment Details {self.experiment_name}")
            console.print(f"> from experiment with {self.model_type}")
            console.print(f"> on {self.date_experiment}")

    def scoreboard(self, file: str = "scoreboard.csv", base_dir: str = None):
        head_scoreboard = [
            "Id",
            "Date",
            "Sample",
            "PDF Type",
            "Target Type",
            "Target Params",
            "Model Type",
            "R2 Score",
            "Train Epoch",
            "Batch Size",
            "Learning Rate",
            "N Layer",
            "N Neurons",
            "Loss Type",
            "Dimension",
            "Max Error",
            "MSE Score",
            "EVS Score",
            "ISE Score",
            "KL Score",
            "Title",
            "PDF params",
            "Model params",
            "Target params",
            "Seed",
        ]

        if base_dir is None:
            base_dir = self.base_dir

        # Relativo solo alle MLP
        target_params_specific = "None"
        if self.model_type in ["GNN", "PNN"]:
            n_layer = len(self.model_params["hidden_layer"])
            n_neurons = 0  # qui meglio fare una funzione
            for layer in self.model_params["hidden_layer"]:
                n_neurons += layer[0]
            epoch = self.train_params["epochs"]
            batch_size = self.train_params["batch_size"]
            loss_type = self.train_params["loss_type"]
            learning_rate = self.train_params["learning_rate"]

            if self.target_type == "GMM":
                target_params_specific = f"{self.target_params['init_params']} C{self.target_params['n_components']} "
            elif self.target_type == "PARZEN":
                target_params_specific = f"H{self.target_params['h']}"

        else:
            n_layer = None
            n_neurons = None
            epoch = None
            batch_size = None
            loss_type = None
            learning_rate = None

        # Questo c'è sempre
        pdf_type = self.pdf.name
        n_samples = self.pdf.n_samples_training  # del training
        pdf_param = self.pdf.params
        dimension = len(self.pdf.params)

        # scoring, questi posso prendermeli da self.save_metrics
        r2_score = self.model_metrics.get("r2")
        max_error_score = self.model_metrics.get("max_error")
        mse_score = self.model_metrics.get("mse")
        evs_score = self.model_metrics.get("evs")
        ise_score = self.model_metrics.get("ise")
        k1_score = self.model_metrics.get("kl")

        write_csv(
            log_name_file=file,
            base_dir=base_dir,
            check_colomn="id",
            head=head_scoreboard,
            id=str(self.id_experiment),
            date=self.date_experiment,
            sample=n_samples,
            pdf_type=pdf_type,
            target_type=self.target_type,
            target_params=target_params_specific,
            model_type=self.model_type,
            r2_score=r2_score,
            train_epoch=epoch,
            batch_size=batch_size,
            learning_rate=learning_rate,
            n_layer=n_layer,
            n_neurons=n_neurons,
            loss_type=loss_type,
            dimension=dimension,
            max_error_score=max_error_score,
            mse_score=mse_score,
            evs_score=evs_score,
            ise_score=ise_score,
            k1_score=k1_score,
            title=self.experiment_name,
            pdf_param=pdf_param,
            model_param=self.model_params,
            target_param=self.target_params,
            seed=self.seed,
        )

        # scrivo nel file solo se è una MLP
        if self.model_type in ["GNN", "PNN"]:

            file_name = file.split(".csv")[0]
            file_name += f"_MLP_{self.pdf.n_samples_training}.csv"
            write_csv(
                log_name_file=file_name,
                base_dir=base_dir,
                check_colomn="id",
                id=str(self.id_experiment),
                date=self.date_experiment,
                pdf_type=pdf_type,
                target_type=self.target_type,
                target_params=target_params_specific,
                model_type=self.model_type,
                r2_score=r2_score,
                train_epoch=epoch,
                batch_size=batch_size,
                learning_rate=learning_rate,
                n_layer=n_layer,
                n_neurons=n_neurons,
                loss_type=loss_type,
                dimension=dimension,
                max_error_score=max_error_score,
                k1_score=k1_score,
                seed=self.seed,
            )
        else:
            file_name = file.split(".csv")[0]
            file_name += f"_statistics_{self.pdf.n_samples_training}.csv"
            write_csv(
                log_name_file=file_name,
                base_dir=base_dir,
                check_colomn="id",
                id=str(self.id_experiment),
                date=self.date_experiment,
                pdf_type=pdf_type,
                model_type=self.model_type,
                r2_score=r2_score,
                dimension=dimension,
                max_error_score=max_error_score,
                mse_score=mse_score,
                evs_score=evs_score,
                ise_score=ise_score,
                k1_score=k1_score,
                model_param=self.model_params,
                seed=self.seed,
            )

    def log_target(
        self,
    ):
        try:
            if self.dataset_params.get("target_type"):
                with open(self.filename_summary_md_path, "a") as file:
                    console = Console(file=file)
                    console.print("## Target")

                    if self.target_params.items():
                        console.print(f"- Using {self.model_type} Target")
                        console.print(
                            f"<details><summary>All Params used in the model for generate the target for the MLP </summary>\n"
                        )

                        console_table_dict(console, self.target_params)
                        console.print("</details>")
                        console.print("")

        except:
            print(
                "error in log_target, trying to log target but no target was provided"
            )

    def log_train_params(self):

        with open(self.filename_summary_md_path, "a") as file:

            if self.train_params != None:
                console = Console(file=file)
                console.print("## Training")
                console.print(
                    f"<details><summary>All Params used for the training </summary>\n"
                )
                console_table_dict(console, self.train_params)
                console.print("</details>")
                console.print("")

    def log_model(self, model=None):
        with open(self.filename_summary_md_path, "a") as file:
            console = Console(file=file)
            console.print("## Model")
            console.print(f"> using model {self.model_type}")
            console.print(f"#### Model Params:")

            if self.model_params.items():
                console.print(
                    f"<details><summary>All Params used in the model </summary>\n"
                )
                console_table_dict(console, self.model_params)
                console.print("</details>")
                console.print("")

            if model is not None:
                console.print(f"<details><summary>Model Architecture </summary>\n")

                console.print(str(model))
                console.print("</details>")
                console.print("")

    def plot_pdf(
        self,
        X_train,
        Y_target,
        X_test,
        Y_test,
        Y_pred,
        bins=32,
        density=True,
        save=True,
        show=False,
        name="pdf",
    ):

        fig, ax = plt.subplots()
        # Plot delle pdf

        if Y_pred is not None:
            ax.plot(X_test, Y_pred, label=self.model_type, color="red")

        if Y_test is not None:
            ax.plot(X_test, Y_test, label="True PDF", color="green", linestyle="--")

        if Y_target is not None:
            ax.scatter(X_train, Y_target, label=f"Target ({self.model_type})")

        ax.hist(
            X_train,
            bins=32,
            density=density,
            alpha=0.5,
            label="Data",
            color="dimgray",
        )
        ax.set_title(f"PDF estimation with {len(X_train)} samples")
        ax.set_xlabel("X")
        ax.set_ylabel("Probability Density")
        ax.legend()

        if save == True:
            extension = ".png"

            img_folder_name = os.path.join(name + "_" + str(self.id_experiment))
            plt.savefig(os.path.join(self.experiment_dir, img_folder_name + extension))
            with open(self.filename_summary_md_path, "a") as file:
                console = Console(file=file)
                console.print("## Plot Prediction")
                console.print("")
                console.print(f'<img src="{img_folder_name + extension}">')
                console.print("")

        if show == True:
            plt.show()

        plt.close(fig)

    def plot_loss(
        self,
        train_loss,
        val_loss: list = None,
        val_acc: list = None,
        loss_name: str = "Loss",
        fig_name: str = "loss",
        save=True,
        show=False,
    ):

        if train_loss is not None:
            fig, ax = plt.subplots()

            sns.lineplot(data=train_loss, label="train loss", color="orange", ax=ax)

            if val_loss is not None:
                sns.lineplot(data=val_loss, label="val loss", color="blue", ax=ax)

            if val_acc is not None:
                sns.lineplot(data=val_acc, label="val R2", ax=ax)

            ax.set_title(f"{loss_name} Plot")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.legend()
            if save == True:
                extension = ".png"
                img_folder_path = self.experiment_dir
                img_folder_name = fig_name + "_" + str(self.id_experiment)
                plt.savefig(os.path.join(img_folder_path, img_folder_name + extension))
                with open(self.filename_summary_md_path, "a") as file:
                    console = Console(file=file)
                    console.print("## Loss Plot")
                    console.print("")
                    console.print(f'<img src="{ img_folder_name + extension}">')
                    console.print("")

            if show == True:
                plt.show()

            # Chiusura della figura
            plt.close(fig)

    def calculate_metrics(
        self,
        y_true_train,
        y_true_test,
        y_predected,
        y_target: list = None,
    ):
        metrics = []
        self.model_metrics = calculate_metrics(y_true_test, y_predected, 4)
        if y_target is not None:
            target_metrics = calculate_metrics(y_true_train, y_target)
            metrics.append({"type": "Target", **target_metrics})
        metrics.append({"type": "Model", **self.model_metrics})

        with open(self.filename_summary_md_path, "a") as file:
            console = Console(file=file)
            console.print("## Metrics:")
            console_table_list_of_dict(console, metrics)

    def log_dataset(
        self,
        notes: str = "",
    ):
        with open(self.filename_summary_md_path, "a") as file:
            console = Console(file=file)
            params_dataset = {}
            console.print("## Dataset")
            console.print("")
            if self.pdf.default is not None:
                summary = f"PDF set as default <b>{self.pdf.default}</b>"
            else:
                summary = "PDF attribute"

            console.print(f"<details><summary>{summary}</summary>\n")

            for dim, dimension_params in enumerate(self.pdf.params):
                console.print(f"#### Dimension {dim+1}")
                console_table_list_of_dict(console, dimension_params)
            console.print("</details>")

            params_dataset["dimension"] = len(self.pdf.params)
            params_dataset["seed"] = self.seed
            params_dataset["n_samples_training"] = self.pdf.n_samples_training
            params_dataset["n_samples_test"] = self.pdf.n_samples_test
            params_dataset["n_samples_val"] = self.pdf.n_samples_validation

            params_dataset["notes"] = notes
            console_table_dict(console, params_dataset)

    def save_example_data(self, console: Console, example_data, details=True):
        console.print("## Example data")
        summary = "Example Data from QM9 dataset Padded"
        console.print(f"<details><summary>{summary}</summary>\n")
        for key, value in example_data.items():

            if isinstance(value, int) or isinstance(value, str):
                console.print(f"#### {key} :\n - {str(value)}")
                if key == "smiles":
                    mol_filepath = os.path.join(
                        self.directory_base, "example_molecule.png"
                    )
                    console.print("\n<img src='example_molecule.png'>")

            else:
                console.print("#### " + key + " :")
                console.print("> __SHAPE__ : " + str(value.shape))
                # print(value.tolist())
                table = console_matrix(console, value.tolist())
        console.print("</details>")


if __name__ == "__main__":
    with open("prova.md", "w") as file:
        console = Console(file=file)

        table = Table(show_header=True, header_style="bold magenta", box=box.MARKDOWN)
        table.add_column("Released", justify="center", style="cyan", no_wrap=True)
        table.add_column("Title", justify="center", style="magenta")

        table.add_row(
            "Dec 20, 2019", "Star Wars: The Rise of Skywalker", "$952,110,690"
        )
        table.add_row("May 25, 2018", "Solo: A Star Wars Story", "$393,151,347")
        table.add_row(
            "Dec 15, 2017", "Star Wars Ep. V111: The Last Jedi", "$1,332,539,889"
        )
        for i in range(3):
            table.add_row("Dec 16, 2016", "Rogue One: A Star Wars Story", f"{i*100}")

        console.print("## INFORMATION", style="blue")
        console.print(table)
        console.print("## DANGER!", style="red on white")
        console.print_json(data={"prova": "proav1"})
        lista = ["prova1", "prova2", "prova3"]
        console_bullet_list(console, lista)

        dicta = {"prova1": "prova1", "prova2": "prova2", "prova3": "prova3"}
        console_table_dict(console, dicta)

        matrix = [["ppp", "bbbb", "zzzz"], ["ppp", "bbbb"], ["ppp", "bbbb"]]
