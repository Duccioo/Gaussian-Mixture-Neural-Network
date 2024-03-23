import optuna
from optuna.trial import TrialState
import torch
import tempfile

import sys
import os


# ---
# a = sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.optuna_model import objective_MLP_allin_gmm


def start_optuna_mlp():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    params = {
        # TRAIN PARAMS:
        "optimizer": ("Adam", "RMSprop"),
        "learning_rate": (1e-5, 1e-1),
        "epoch": (100, 1000),
        "batch_size": (2, 64),
        # MODEL PARAMS:
        "n_layers": (1, 4),
        "range_neurons": (4, 64),
        "range_dropout": (0.00, 0.2),
        "suggest_activation": ("relu", "tanh"),
        "suggest_last_activation": ("lambda", None),
        # GMM and DATASET PARAMS:
        "n_samples": 100,
        "n_components": (2, 15),
        "init_params_gmm": ("k-means++", "kmeans"),
        "seed": 42,
        "n_init": 60,
        "max_iter": 80,
    }

    # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    optuna.logging.get_logger("optuna")
    study_name = f"BEST GMM MLP 2"  # Unique identifier of the study.
    storage_name = "sqlite:///optuna-02.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
    )

    study.set_metric_names(["R2 score"])

    tmp_dir = tempfile.TemporaryDirectory()

    study.optimize(
        lambda trial: objective_MLP_allin_gmm(trial, params, tmp_dir.name),
        n_trials=300,
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
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))



if __name__ == "__main__":
    start_optuna_mlp()