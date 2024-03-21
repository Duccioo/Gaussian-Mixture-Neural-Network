import optuna
from optuna.trial import TrialState

import torch
import numpy as np

from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture

from rich.progress import track

# ---
from utils.data_manager import PDF


def generate_target_MLP_optuna(
    trial: optuna.Trial,
    gm_model: GaussianMixture,
    X,
    Y_test,
    progress_bar: bool = False,
):
    # try to load the target data:

    Y = []
    # print(X.shape)
    pb = (
        track(enumerate(X), description="Generating Target: ", total=len(X))
        if progress_bar
        else enumerate(X)
    )

    for indx, sample in pb:

        X_1 = np.delete(X, indx, axis=0)
        gm_model.fit(X_1)
        Y.append(np.exp(gm_model.score_samples(sample.reshape(-1, X_1.shape[1]))))

        if indx > 5:
            r2_value = r2_score(np.array(Y).reshape(-1, 1), Y_test[0 : len(Y)])

            trial.report(r2_value, indx)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    Y = np.array(Y).reshape(-1, 1)

    return X, Y


def get_dataloader(trial: optuna.Trial, X_train, Y_train, batch_size: list = [2, 64]):

    batch_size = trial.suggest_int("batch_size", batch_size[0], batch_size[1])

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)

    xy_train = torch.cat((X_train, Y_train), 1)

    train_loader = torch.utils.data.DataLoader(
        xy_train,
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader


def objective_gmm_target(trial: optuna.Trial, params, device):

    pdf = PDF(default="MULTIVARIATE_1254")
    stepper_x_test = 0.01
    bias = False

    n_components = trial.suggest_int(
        "n_components", params["n_components"][0], params["n_components"][1]
    )

    init_params_gmm = trial.suggest_categorical(
        "init_params_gmm", params["init_params_gmm"]
    )

    seed = trial.suggest_int("seed", params["seed"][0], params["seed"][1])
    n_init = trial.suggest_int("n_init", params["n_init"][0], params["n_init"][1])
    max_iter = trial.suggest_int(
        "max_iter", params["max_iter"][0], params["max_iter"][1]
    )
    n_samples = trial.suggest_int(
        "n_samples", params["n_samples"][0], params["n_samples"][1]
    )

    gm_model = GaussianMixture(
        n_components=n_components,
        init_params=init_params_gmm,
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
    )

    X_train, Y_train = pdf.generate_training(n_samples=n_samples, seed=seed)

    # if trial.should_prune():
    #     raise optuna.exceptions.TrialPruned()

    _, gmm_target_y = generate_target_MLP_optuna(
        trial, gm_model=gm_model, X=X_train, Y_test=Y_train, progress_bar=True
    )
    # Validation of the model.

    r2_value = r2_score(Y_train, gmm_target_y)

    return r2_value


if __name__ == "__main__":

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    params = {
        # TRAIN PARAMS:
        "n_samples": (100, 400),
        "n_components": (4, 20),
        "init_params_gmm": ("kmeans", "k-means++"),
        "seed": (0, 100),
        "n_init": (10, 100),
        "max_iter": (10, 100),
    }

    optuna.logging.get_logger("optuna")
    study_name = f"perfect GMM Target 4"  # Unique identifier of the study.
    storage_name = "sqlite:///test_v2_3.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
    )

    study.set_metric_names(["R2 score"])

    study.optimize(
        lambda trial: objective_gmm_target(trial, params, DEVICE),
        n_trials=300,
        timeout=None,
        n_jobs=1,
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
