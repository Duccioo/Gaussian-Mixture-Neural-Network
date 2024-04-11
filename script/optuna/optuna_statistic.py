from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture
import optuna
from optuna.trial import TrialState
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import argparse

import sys
import os

# ---
a = sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.knn_model import KNN_Model
from model.parzen_model import ParzenWindow_Model
from utils.data_manager import PDF


def objective_knn(trial: optuna.Trial, params):

    if isinstance(params["n_samples"], (list, tuple)):
        n_samples = trial.suggest_int(
            "n_samples", params["n_samples"][0], params["n_samples"][1]
        )
    else:
        n_samples = params["n_samples"]

    if isinstance(params["dataset_seed"], (list, tuple)):
        dataset_seed = trial.suggest_int(
            "dataset_seed", params["dataset_seed"][0], params["dataset_seed"][1]
        )
    else:
        dataset_seed = params["dataset_seed"]

    if isinstance(params["k1"], (list, tuple)):
        knn_k1 = trial.suggest_float("k1", params["k1"][0], params["k1"][1])
    else:
        knn_k1 = params["k1"]

    if isinstance(params["kn"], (list, tuple)):
        knn_kn = trial.suggest_int("kn", params["kn"][0], params["kn"][1])
    else:
        knn_kn = params["kn"]

    X_train, _, X_test, Y_test = load_dataset(
        n_samples, dataset_seed, type=params["dataset_type"]
    )

    model_parzen = KNN_Model(knn_k1, knn_kn)
    model_parzen.fit(training=X_train)
    pdf_predicted_parzen = model_parzen.predict(test=X_test)

    r2_value = r2_score(Y_test, pdf_predicted_parzen)

    trial.report(r2_value, 0)

    return r2_value


def objective_gmm(trial: optuna.Trial, params):

    if isinstance(params["n_samples"], (list, tuple)):
        n_samples = trial.suggest_int(
            "n_samples", params["n_samples"][0], params["n_samples"][1]
        )
    else:
        n_samples = params["n_samples"]

    if isinstance(params["dataset_seed"], (list, tuple)):
        dataset_seed = trial.suggest_int(
            "dataset_seed", params["dataset_seed"][0], params["dataset_seed"][1]
        )
    else:
        dataset_seed = params["dataset_seed"]

    if isinstance(params["gmm_seed"], (list, tuple)):
        gmm_seed = trial.suggest_int(
            "gmm_seed", params["gmm_seed"][0], params["gmm_seed"][1]
        )
    else:
        gmm_seed = params["gmm_seed"]

    if isinstance(params["n_components"], (list, tuple)):
        n_components = trial.suggest_int(
            "n_components", params["n_components"][0], params["n_components"][1]
        )
    else:
        n_components = params["n_components"]

    if isinstance(params["init_params"], (list, tuple)):
        init_param_gmm = trial.suggest_categorical("init_params", params["init_params"])
    else:
        init_param_gmm = params["init_params"]

    if isinstance(params["max_iter"], (list, tuple)):
        max_iter = trial.suggest_int(
            "max_iter", params["max_iter"][0], params["max_iter"][1], step=10
        )
    else:
        max_iter = params["max_iter"]

    if isinstance(params["n_init"], (list, tuple)):
        n_init = trial.suggest_int(
            "n_init", params["n_init"][0], params["n_init"][1], step=10
        )
    else:
        n_init = params["n_init"]

    X_train, _, X_test, Y_test = load_dataset(
        n_samples, dataset_seed, type=params["dataset_type"]
    )

    # train the GMM model
    model_gmm = GaussianMixture(
        n_components=n_components,
        random_state=gmm_seed,
        init_params=init_param_gmm,
        max_iter=max_iter,
        n_init=n_init,
    )
    model_gmm.fit(X_train)
    # predict the pdf with GMM
    pdf_predicted = np.exp(model_gmm.score_samples(X_test))

    r2_value = r2_score(Y_test, pdf_predicted)

    trial.report(r2_value, 0)

    return r2_value


def objective_parzen(trial: optuna.Trial, params):

    if isinstance(params["n_samples"], (list, tuple)):
        n_samples = trial.suggest_int(
            "n_samples", params["n_samples"][0], params["n_samples"][1]
        )
    else:
        n_samples = params["n_samples"]

    if isinstance(params["dataset_seed"], (list, tuple)):
        dataset_seed = trial.suggest_int(
            "dataset_seed", params["dataset_seed"][0], params["dataset_seed"][1]
        )
    else:
        dataset_seed = params["dataset_seed"]

    if isinstance(params["h"], (list, tuple)):
        h_ = trial.suggest_float("h", params["h"][0], params["h"][1])
    else:
        h_ = params["h"]

    X_train, _, X_test, Y_test = load_dataset(
        n_samples, dataset_seed, type=params["dataset_type"]
    )

    model_parzen = ParzenWindow_Model(h=h_)
    model_parzen.fit(training=X_train)

    pdf_predicted_parzen = model_parzen.predict(test=X_test)

    r2_value = r2_score(Y_test, pdf_predicted_parzen)

    trial.report(r2_value, 0)

    return r2_value


def load_dataset(n_samples: int = 100, seed: int = 42, type: str = "multivariate"):
    if type.lower() in ["multivariate", "multivariate_1254"]:
        pdf = PDF(default="MULTIVARIATE_1254")
    elif type.lower() in ["exponential", "exp"]:
        pdf = PDF(default="EXPONENTIAL_06")

    stepper_x_test = 0.01
    X_train, Y_train = pdf.generate_training(n_samples=n_samples)

    # generate the data for plotting the pdf
    X_test, Y_test = pdf.generate_test(stepper=stepper_x_test)

    return X_train, Y_train, X_test, Y_test


def arg_parsing():
    # command line parsing
    parser = argparse.ArgumentParser(description="Project for AI Exam")
    parser.add_argument(
        "--dataset", type=str, default="multivariate"
    )  # multivariate or exp
    parser.add_argument(
        "--objective", type=str, default="objective_parzen"
    )  # knn, parzen, gmm
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--trials", type=int, default=500)
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = arg_parsing()

    params = {
        # KNN
        "k1": [0.1, 10],
        "kn": [1, 100],
        # PARZEN
        "h": [0.01, 1.0],
        # GMM
        "init_params": ["k-means++", "kmeans", "random", "random_from_data"],
        "n_init": [10, 100],
        "max_iter": [10, 100],
        "n_components": [1, 32],
        "gmm_seed": [1, 100],
        # DATASET
        "n_samples": args.samples,
        "dataset_seed": None,
        "dataset_type": args.dataset,  # multivariate or exp
        # OBJECTIVE
        "objective_name": args.objective,  # knn, gmm, parzen
        "n_trials": args.trials,
        "save_database": True,  # save the optuna database
    }

    # optuna.logging.get_logger("optuna")
    study_name = f"{params['objective_name']} {params['dataset_type']} {params['n_samples']}"  # Unique identifier of the study.
    storage_name = f"db02-statistic_{params['n_samples']}_{params['dataset_type']}.db"

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

    objective = globals()["objective_" + params["objective_name"]]

    study.optimize(
        lambda trial: objective(trial, params),
        n_trials=params["n_trials"],
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

    # X_train, Y_train, X_test, Y_test = load_dataset(
    #     n_samples=params["n_samples"], type="exponential"
    # )

    # model_best = ParzenWindow_Model(h=trial.params["h"])
    # model_best.fit(training=X_train)

    # Y_predicted = model_best.predict(X_test)

    # print(
    #     "R2 score: ",
    #     r2_score(Y_test, Y_predicted),
    # )

    # sns.lineplot(x=X_test.flatten(), y=Y_test.flatten(), color="green", label="True")
    # sns.lineplot(
    #     x=X_test.flatten(),
    #     y=Y_predicted,
    #     color="red",
    #     label="Predicted",
    # )
    # plt.hist(X_train, bins=32, density=True, alpha=0.7, color="grey")
    # plt.legend()
    # plt.show()
