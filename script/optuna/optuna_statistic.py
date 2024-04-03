from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture
import optuna
from optuna.trial import TrialState
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


# ---
from model.knn_model import KNN_Model
from model.parzen_model import ParzenWindow_Model
from utils.data_manager import PDF


def objective_knn(trial: optuna.Trial, params):

    if isinstance(params["n_samples"], (list, tuple)):
        n_samples = trial.suggest_int("n_samples", params["n_samples"][0], params["n_samples"][1])
    else:
        n_samples = params["n_samples"]

    if isinstance(params["seed"], (list, tuple)):
        seed = trial.suggest_int("seed", params["seed"][0], params["seed"][1])
    else:
        seed = params["seed"]

    if isinstance(params["k1"], (list, tuple)):
        knn_k1 = trial.suggest_float("k1", params["k1"][0], params["k1"][1])
    else:
        knn_k1 = params["k1"]

    if isinstance(params["kn"], (list, tuple)):
        knn_kn = trial.suggest_int("kn", params["kn"][0], params["kn"][1])
    else:
        knn_kn = params["kn"]

    X_train, _, X_test, Y_test = load_dataset(n_samples, seed, type=params["dataset_type"])

    model_parzen = KNN_Model(knn_k1, knn_kn)
    model_parzen.fit(training=X_train)
    pdf_predicted_parzen = model_parzen.predict(test=X_test)

    r2_value = r2_score(pdf_predicted_parzen, Y_test)

    trial.report(r2_value, 0)

    return r2_value


def objective_gmm(trial: optuna.Trial, params, X_train, X_test, Y_test):

    if isinstance(params["n_samples"], (list, tuple)):
        n_samples = trial.suggest_int("n_samples", params["n_samples"][0], params["n_samples"][1])
    else:
        n_samples = params["n_samples"]

    if isinstance(params["seed"], (list, tuple)):
        seed = trial.suggest_int("seed", params["seed"][0], params["seed"][1])
    else:
        seed = params["seed"]

    if isinstance(params["n_components"], (list, tuple)):
        n_components = trial.suggest_int("n_components", params["n_components"][0], params["n_components"][1])
    else:
        n_components = params["n_components"]

    if isinstance(params["init_params"], (list, tuple)):
        init_param_gmm = trial.suggest_categorical("init_params", params["init_params"])
    else:
        init_param_gmm = params["init_params"]

    if isinstance(params["max_iter"], (list, tuple)):
        max_iter = trial.suggest_int("max_iter", params["max_iter"][0], params["max_iter"][1], step=10)
    else:
        max_iter = params["max_iter"]

    if isinstance(params["n_init"], (list, tuple)):
        n_init = trial.suggest_int("n_init", params["n_init"][0], params["n_init"][1], step=10)
    else:
        n_init = params["n_init"]

    X_train, _, X_test, Y_test = load_dataset(n_samples, seed, type=params["dataset_type"])

    # train the GMM model
    model_gmm = GaussianMixture(
        n_components=n_components,
        random_state=seed,
        init_params=init_param_gmm,
        max_iter=max_iter,
        n_init=n_init,
    )
    model_gmm.fit(X_train)
    # predict the pdf with GMM
    pdf_predicted = np.exp(model_gmm.score_samples(X_test))

    r2_value = r2_score(pdf_predicted, Y_test)

    trial.report(r2_value, 0)

    return r2_value


def objective_parzen(trial: optuna.Trial, params, X_train, X_test, Y_test):

    if isinstance(params["n_samples"], (list, tuple)):
        n_samples = trial.suggest_int("n_samples", params["n_samples"][0], params["n_samples"][1])
    else:
        n_samples = params["n_samples"]

    if isinstance(params["seed"], (list, tuple)):
        seed = trial.suggest_int("seed", params["seed"][0], params["seed"][1])
    else:
        seed = params["seed"]

    if isinstance(params["h"], (list, tuple)):
        h_ = trial.suggest_float("h", params["h"][0], params["h"][1])
    else:
        h_ = params["h"]

    X_train, _, X_test, Y_test = load_dataset(n_samples, seed, type=params["dataset_type"])

    model_parzen = ParzenWindow_Model(h=h_)
    model_parzen.fit(training=X_train)

    pdf_predicted_parzen = model_parzen.predict(test=X_test)

    r2_value = r2_score(pdf_predicted_parzen, Y_test)

    trial.report(r2_value, 0)

    return r2_value


def load_dataset(n_samples: int = 100, seed: int = 42, type: str = "multivariate"):
    if type.lower() in ["multivariate", "multivariate_1254"]:
        pdf = PDF(default="MULTIVARIATE_1254")
    elif type.lower() in ["exponential", "exp"]:
        pdf = PDF([[{"type": "exponential", "rate": 0.6}]], name="exponential 0.6")

    stepper_x_test = 0.01
    X_train, Y_train = pdf.generate_training(n_samples=n_samples, seed=seed)

    # generate the data for plotting the pdf
    X_test, Y_test = pdf.generate_test(stepper=stepper_x_test)

    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":

    params = {
        # KNN
        "k1": [0.1, 10],
        "kn": [1, 100],
        # PARZEN
        "h": [0.1, 1.0],
        # GMM
        "init_params": ["k-means++", "kmeans", "random", "random_from_data"],
        "n_init": [10, 100],
        "max_iter": [10, 100],
        # DATASET
        "n_samples": 100,
        "seed": [1, 100],
        "dataset_type": "multivariate",  # multivariate or exp
        # OBJECTIVE
        "objective_name": "objective_knn",  # objective_knn, objective_gmm, objective_parzen
    }

    # optuna.logging.get_logger("optuna")
    study_name = f"{params['objective_name']} {params['dataset_type']} {params['n_samples']}"  # Unique identifier of the study.
    storage_name = "sqlite:///optuna-statistic.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
    )

    study.set_metric_names(["R2 score"])

    objective = globals()[params["objective_name"]]

    study.optimize(
        lambda trial: objective(trial, params),
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
    # K1 = 1.4461890579977732
    # KN = 100
    # n_samples = 386
    # seed = 42
    # X_train, Y_train, X_test, Y_test = load_dataset(n_samples=n_samples, seed=seed, type="exponential")

    # model_knn = KNN_Model(k1=K1, kn=KN)
    # model_knn.fit(training=X_train)
    # pdf_predicted = model_knn.predict(test=X_test)
    # r2_value = r2_score(pdf_predicted, Y_test)
    # print(r2_value)

    # sns.lineplot(x=X_test.flatten(), y=Y_test.flatten(), color="green", label="True")
    # sns.lineplot(x=X_test.flatten(), y=pdf_predicted, label="base", color="red")
    # plt.legend()
