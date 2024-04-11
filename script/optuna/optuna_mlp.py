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
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        "n_components": (2, 40),
        "init_params_gmm": ("k-means++", "kmeans"),
        "n_init": (10, 100),
        "max_iter": (10, 100),
        "gmm_seed": (0, 100),
        # PARZEN PARAMS:
        "h": (0.001, 1),
        # DATASET PARAMS:
        "dataset_type": "exp",  # multivariate or exp
        "n_samples": 100,
        # -- seed che influisce sui pesi della MLP
        "seed": (0, 100),
        "target_type": "GMM",  # GMM or PARZEN
        "pruning": True,  # use pruning if True
        "trials": 500,
        "save_database": True,  # save study in database
    }

    # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    optuna.logging.get_logger("optuna")
    study_name = (
        f"MLP {params['dataset_type']} {params['target_type']} {params['n_samples']}"
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
