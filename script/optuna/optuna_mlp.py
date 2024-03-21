import optuna
from optuna.trial import TrialState
import torch

# ---
from model.optuna_model import objective_MLP_allin_gmm


if __name__ == "__main__":

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
        "range_dropout": (0.01, 0.17),
        "suggest_activation": ("relu", "tanh"),
        "suggest_last_activation": ("lambda", None),
        # GMM and DATASET PARAMS:
        "n_samples": 400,
        "n_components": (5, 10),
        "init_params_gmm": ("k-means++", "kmeans"),
        "seed": 42,
        "n_init": (10, 100),
        "max_iter": (10, 100),
    }

    # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    optuna.logging.get_logger("optuna")
    study_name = f"BEST GMM MLP 1"  # Unique identifier of the study.
    storage_name = "sqlite:///optuna-02.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
    )

    study.set_metric_names(["R2 score"])

    study.optimize(
        lambda trial: objective_MLP_allin_gmm(trial, params),
        n_trials=300,
        timeout=None,
        n_jobs=2,
        show_progress_bar=True,
    )

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
