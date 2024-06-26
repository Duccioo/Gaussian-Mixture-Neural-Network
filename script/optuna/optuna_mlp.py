import optuna
from optuna.trial import TrialState
import torch
import tempfile

import sys
import os
import argparse


# ---
a = sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.optuna_model import objective_MLP_allin_gmm


def arg_parsing():
    # command line parsing
    parser = argparse.ArgumentParser(description="Project for AI Exam")
    parser.add_argument("--dataset", type=str, default="multivariate")  # multivariate or exp
    parser.add_argument("--objective", type=str, default="PARZEN")  # PARZEN or GMM (target)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--trials", type=int, default=500)
    args = parser.parse_args()

    return args


def start_optuna_mlp():

    args = arg_parsing()

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
        "init_params_gmm": ("k-means++", "kmeans"),
        "n_init": (10, 100),
        "max_iter": (10, 100),
        "gmm_seed": (0, 100),
        # PARZEN PARAMS:
        "h": (0.001, 1),
        # DATASET PARAMS:
        "dataset_type": args.dataset,  # multivariate or exp
        "n_samples": args.samples,
        "seed": (0, 100),  # seed che influisce sui pesi della MLP
        "target_type": args.objective,  # GMM or PARZEN
        "pruning": False,  # use pruning if True
        "trials": args.trials,
        "save_database": True,  # save study in database
    }

    # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    optuna.logging.get_logger("optuna")
    study_name = (
        f"MLP {params['dataset_type']} {params['target_type']} {params['n_samples']} (no pruning) COR 3"
    )
    if not isinstance(params["seed"], (list, tuple)):
        study_name += f" fixed {params['seed']} seed"

    storage_name = f"db02-MLP_{params['n_samples']}_{params['dataset_type']}.db"

    db_folder = "optuna_database"
    storage_path = db_folder + "/" + storage_name
    storage_path = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path if params["save_database"] else None,
        direction="maximize",
        load_if_exists=True,
    )

    study.set_metric_names(["R2 score"])

    tmp_dir = tempfile.TemporaryDirectory()

    study.optimize(
        lambda trial: objective_MLP_allin_gmm(trial, params, tmp_dir.name, device),
        n_trials=params["trials"],
        timeout=None,
        n_jobs=1,
        show_progress_bar=True,
    )

    tmp_dir.cleanup()

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trials

    print("  Value: ", trial[0].value)

    print("  Params: ")
    for key, value in trial[0].params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    start_optuna_mlp()
