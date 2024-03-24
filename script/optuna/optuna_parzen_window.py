import os

from matplotlib import pyplot as plt
import seaborn as sns


from sklearn.metrics import r2_score
import optuna
from optuna.trial import TrialState


# ---
from model.parzen_model import ParzenWindow_Model
from utils.data_manager import load_multivariate_dataset


def objective_parzen(trial: optuna.Trial, params, X_train, X_test, Y_test):
    parzen_h = trial.suggest_float("h", params["h"][0], params["h"][1])
    n_samples = trial.suggest_int(
        "n_samples", params["n_samples"][0], params["n_samples"][1]
    )
    seed = trial.suggest_int("seed", params["seed"][0], params["seed"][1])
    X_train, Y_train, X_test, Y_test = load_multivariate_dataset(n_samples, seed)

    model_parzen = ParzenWindow_Model(h=parzen_h)
    model_parzen.fit(training=X_train)
    pdf_predicted_parzen = model_parzen.predict(test=X_test)

    r2_value = r2_score(pdf_predicted_parzen, Y_test)

    trial.report(r2_value, 0)

    return r2_value


if __name__ == "__main__":

    params = {"h": [0.1, 1.0], "n_samples": [100, 400], "seed": [1, 100]}
    
    # create the folder for the project:
    experiment_name = "perfect Parzen Window Optuna"

    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    

    # optuna.logging.get_logger("optuna")
    study_name = f"perfect Parzen Window"  # Unique identifier of the study.
    storage_name = "sqlite:///optuna-v0.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
    )

    study.set_metric_names(["R2 score"])

    study.optimize(
        lambda trial: objective_parzen(trial, params, X_train, X_test, Y_test),
        n_trials=300,
        timeout=None,
        n_jobs=-1,
        show_progress_bar=True,
    )

    # ---
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

    # TESTING BEST RESULT

    parzen_h = 0.1689521226813387
    n_samples = 389

    seed = 58
    X_train, Y_train, X_test, Y_test = load_multivariate_dataset(
        n_samples=n_samples, seed=seed
    )

    model_parzen = ParzenWindow_Model(h=parzen_h)
    model_parzen.fit(training=X_train)
    pdf_predicted_parzen = model_parzen.predict(test=X_test)
    r2_value = r2_score(pdf_predicted_parzen, Y_test)
    print(r2_value)

    sns.lineplot(x=X_test.flatten(), y=Y_test.flatten(), color="green", label="True")
    sns.lineplot(x=X_test.flatten(), y=pdf_predicted_parzen, label="base", color="red")
    plt.legend()
