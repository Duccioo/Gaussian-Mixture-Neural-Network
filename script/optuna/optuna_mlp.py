import optuna
from optuna.trial import TrialState
import torch
import tempfile

import sys
import os


# ---
a = sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.optuna_model import objective_MLP_allin_gmm


def start_optuna_mlp():
    device = "cpu"

    params = {
        # TRAIN PARAMS:
        "optimizer": ("Adam", "RMSprop"),
        "learning_rate": (1e-5, 1e-1),
        "epoch": (100, 1000),
        "batch_size": (2, 80),
        "loss": ("mse_loss", "huber_loss"),
        # MODEL PARAMS:
        "n_layers": (1, 5),
        "range_neurons": (4, 64),
        "range_dropout": 0,
        "suggest_activation": ("relu", "tanh", "sigmoid"),
        "suggest_last_activation": ("lambda", None),
        # GMM PARAMS:
        "n_components": (2, 15),
        "init_params_gmm": ("k-means++", "kmeans"),
        "n_init": (10, 100),
        "max_iter": (10, 100),
        # PARZEN PARAMS:
        "h": (0.001, 1),
        # DATASET PARAMS:
        "dataset_type": "exp",  # multivariate or exp
        "n_samples": 100,
        "seed": 42,
        "target_type": "GMM",
    }

    # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    optuna.logging.get_logger("optuna")
    study_name = f"{params['dataset_type']} {params['target_type']} MLP {params['n_samples']} fixed 42 seed (1)"  # Unique identifier of the study.
    storage_name = f"sqlite:///optuna-{params['n_samples']} .db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
    )

    study.set_metric_names(["R2 score"])

    tmp_dir = tempfile.TemporaryDirectory()

    study.optimize(
        lambda trial: objective_MLP_allin_gmm(trial, params, tmp_dir.name, device),
        n_trials=1000,
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
