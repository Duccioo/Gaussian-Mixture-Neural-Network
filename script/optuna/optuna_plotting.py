import optuna

import seaborn as sns

import numpy as np

import sys
import os
import argparse
import matplotlib.pyplot as plt
from rich.progress import track


# ---
a = sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import check_base_dir


def get_top_trials(study, n_elements: int = 10):
    valid_trials = [trial for trial in study.trials if trial.values is not None]
    top_trials = sorted(valid_trials, key=lambda x: x.values[0], reverse=True)
    top_trials = top_trials[:n_elements]

    return top_trials


def get_worse_trials(study, n_elements: int = 10):
    valid_trials = [trial for trial in study.trials if trial.values is not None]
    worse_trials = sorted(valid_trials, key=lambda x: x.values[0], reverse=False)
    worse_trials = worse_trials[:n_elements]

    return worse_trials


def select_trials_from_param(trials: list = [], param: str = "learning_rate"):
    X = [trial.params[param] for trial in trials]
    X = [elem if elem is not None else "None" for elem in X]
    R2_score = [trial.values[0] for trial in trials]
    return X, R2_score


if __name__ == "__main__":

    device = "cpu"

    params = {
        # TRAIN PARAMS:
        "optimizer": "Adam",
        "learning_rate": (1e-5, 1e-2),
        "epoch": (100, 1000),
        "batch_size": (2, 80),
        "loss": ("mse_loss", "huber_loss"),
        # MODEL PARAMS:
        "n_layers": (1, 4),
        "range_neurons": (4, 64),
        "range_dropout": 0,
        "suggest_activation": ("relu", "tanh", "sigmoid"),
        "suggest_last_activation": ("lambda", None),
        # GMM PARAMS:
        "n_components": [2, 40],
        "init_params_gmm": ("k-means++", "kmeans", "random", "random_from_data"),
        "n_init": (10, 100),
        "max_iter": (10, 100),
        "gmm_seed": (0, 100),
        # PARZEN PARAMS:
        "h": (0.001, 1),
        # DATASET PARAMS:
        "dataset_type": "multivariate",  # multivariate or exp
        "n_samples": 100,
        "seed": (0, 100),  # seed che influisce sui pesi della MLP
        "target_type": "GMM",  # GMM or PARZEN
        "pruning": False,  # use pruning if True
        "trials": 400,
        "save_database": True,  # save study in database
    }

    optuna.logging.get_logger("optuna")
    study_name = f"MLP {params['dataset_type']} {params['target_type']} {params['n_samples']} (no pruning) COR"
    if not isinstance(params["seed"], (list, tuple)):
        study_name += f" fixed {params['seed']} seed"

    storage_name = f"db02-MLP_{params['n_samples']}_{params['dataset_type']}.db"

    db_folder = "optuna_database"
    storage_path = db_folder + "/" + storage_name
    storage_path = f"sqlite:///{storage_path}"

    study = optuna.load_study(
        study_name=study_name,
        storage=storage_path,
    )

    best_params = [params for params in optuna.importance.get_param_importances(study)][
        :4
    ]
    n_trials = 15  # number of trials to plot

    # controllo se la directory esiste altrimenti la creo:
    directory_optuna_result = check_base_dir(
        "..",
        "..",
        "result",
        "optuna_md",
        f"{params['target_type']}_{params['n_samples']}",
    )
    # creo se non esiste già la cartella delle immagini
    check_base_dir(directory_optuna_result, "img")

    params_to_look = [
        "n_units_l0",
        "epoch",
        "n_layers",
        "learning_rate",
        "activation_l0",
        "last_activation",
        "batch_size",
        "init_params_gmm"
    ]

    if params["target_type"] == "PARZEN":
        params_to_look.append("h")
    else:
        params_to_look.append("n_components")

    print(f"database ->{study_name}<- loaded")

    top_trials = get_top_trials(study, n_trials)
    worse_trials = get_worse_trials(study, n_trials)

    # filename del file markdown
    filename_md = (
        f"{params['target_type']}_{params['n_samples']}_{params['dataset_type']}.md"
    )

    # filename del plot dell'immagine che mostra l'importanza dei parametri
    importance_filename = os.path.join(
        "img",
        f"{params['n_samples']}_{params['dataset_type']}_importance_r2.png",
    )

    # carico e salvo l'immagine che mostra l'importanca dei parametri
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig(os.path.join(directory_optuna_result, importance_filename))

    # apro il file markdown per iniziare a scrivere i risultati di optuna
    with open(os.path.join(directory_optuna_result, filename_md), "w") as f:
        f.write(
            f"## MLP {params['target_type']} {params['n_samples']} {params['dataset_type']}"
        )
        f.write(f"\n\n![result]({importance_filename})")

    # plotting dei parametri più interessanti:
    progress_bar = track(params_to_look, description="Plotting...")

    for param_s in progress_bar:

        X_top, value_top = select_trials_from_param(top_trials, param_s)
        X_worse, value_worse = select_trials_from_param(worse_trials, param_s)

        # Plot del rank
        fig = plt.figure()
        sns.set_theme(style="darkgrid")
        # fig, ax = plt.subplots()
        plt.grid(True)
        plt.xlabel(f"{param_s}")
        plt.ylabel("R2 Score")

        # creo una leggera line verticale dove si trova il valore migliore
        plt.axvline(x=X_top[0], color="green", linestyle="--", alpha=0.1)

        # normalizzo i dati per dare l'intensità del colore in base a quanto fa schifo o è buono un valore
        value_top_normalizzati = (value_top - np.min(value_top)) / (
            np.max(value_top) - np.min(value_top)
        )

        value_worst_normalizzati = 1 - (value_worse - np.min(value_worse)) / (
            np.max(value_worse) - np.min(value_worse)
        )

        sns.scatterplot(
            x=X_top,
            y=value_top,
            alpha=value_top_normalizzati,
            label="best",
            color="green",
        )
        sns.scatterplot(
            x=X_worse,
            y=value_worse,
            alpha=value_worst_normalizzati,
            label="worse",
            color="red",
        )

        plt.legend()
        plt.margins(x=0.2, y=0.1)

        filename = os.path.join(
            directory_optuna_result,
            "img",
            f"{params['n_samples']}_{params['dataset_type']}_{param_s}",
        )
        plt.savefig(filename + "_r2.png")
        # plt.show()
        plt.close(fig)

        # scrivo nel file dei risultati di optuna anche l'immagine appena generata
        with open(os.path.join(directory_optuna_result, filename_md), "a") as f:

            f_m = os.path.join(
                "img", f"{params['n_samples']}_{params['dataset_type']}_{param_s}"
            )

            f.write(f"\n\n![result]({f_m}_r2.png)")

    # scrivo nel file i top 3 e i worst 3 valori e parametri:
    with open(os.path.join(directory_optuna_result, filename_md), "a") as f:
        f.write("\n")
        f.write("### TOP 3\n")
        for i in range(0, 3):
            f.write(f"- R2 score: **{top_trials[i].values[0]}**\n")
            for p, value in top_trials[i].params.items():
                f.write(f"\t - **{p}** : *{value}*\n")

            f.write("\n")

        f.write("\n")
        f.write("### WORST 3\n")
        for i in range(0, 3):
            f.write(f"- R2 score: **{worse_trials[i].values[0]}**\n")
            for p, value in top_trials[i].params.items():
                f.write(f"\t - **{p}** : *{value}*\n")

            f.write("\n")
