from sklearn.metrics import r2_score
import optuna
from optuna.trial import TrialState
from matplotlib import pyplot as plt
import seaborn as sns


# ---
from model.knn_model import KNN_Model
from utils.data_manager import PDF


def objective_knn(trial: optuna.Trial, params):
    knn_k1 = trial.suggest_float("k1", params["k1"][0], params["k1"][1])
    knn_kn = trial.suggest_int("kn", params["kn"][0], params["kn"][1])
    n_samples = trial.suggest_int(
        "n_samples", params["n_samples"][0], params["n_samples"][1]
    )

    seed = trial.suggest_int("seed", params["seed"][0], params["seed"][1])
    X_train, Y_train, X_test, Y_test = load_dataset(n_samples, seed)

    model_parzen = KNN_Model(knn_k1, knn_kn)
    model_parzen.fit(training=X_train)
    pdf_predicted_parzen = model_parzen.predict(test=X_test)

    r2_value = r2_score(pdf_predicted_parzen, Y_test)

    trial.report(r2_value, 0)

    return r2_value


def load_dataset(n_samples, seed):
    pdf = PDF(default="MULTIVARIATE_1254")
    stepper_x_test = 0.01
    X_train, Y_train = pdf.generate_training(n_samples=n_samples, seed=seed)

    # generate the data for plotting the pdf
    X_test, Y_test = pdf.generate_test(stepper=stepper_x_test)

    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":

    params = {
        "k1": [0.1, 10],
        "kn": [1, 100],
        "n_samples": [100, 400],
        "seed": [1, 100],
    }

    # optuna.logging.get_logger("optuna")
    study_name = f"perfect KNN 2"  # Unique identifier of the study.
    storage_name = "sqlite:///optuna-v0.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
    )

    study.set_metric_names(["R2 score"])

    study.optimize(
        lambda trial: objective_knn(trial, params),
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
    K1 = 1.4461890579977732
    KN = 100
    n_samples = 386
    seed = 42
    X_train, Y_train, X_test, Y_test = load_dataset(n_samples=n_samples, seed=seed)

    model_knn = KNN_Model(k1=K1, kn=KN)
    model_knn.fit(training=X_train)
    pdf_predicted = model_knn.predict(test=X_test)
    r2_value = r2_score(pdf_predicted, Y_test)
    print(r2_value)

    sns.lineplot(x=X_test.flatten(), y=Y_test.flatten(), color="green", label="True")
    sns.lineplot(x=X_test.flatten(), y=pdf_predicted, label="base", color="red")
    plt.legend()
