import argparse
import operator

import torch.nn as nn

from sklearn.mixture import GaussianMixture

import seaborn as sns
import matplotlib.pyplot as plt

# ---
from utils.data_manager import PDF
from model.nn_model import NeuralNetworkModular
from model.gm_model import gen_target_with_gm_parallel
from model.parzen_model import gen_target_with_parzen_parallel, ParzenWindow_Model
from utils.utils import set_seed
from utils.summary import Summary
from training import training, evaluation


def arg_parsing():
    # command line parsing
    parser = argparse.ArgumentParser(description="Project for AI Exam")
    parser.add_argument("--pdf", type=str, default="default")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=100)

    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)

    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--mlp_targets", action="store_true", default=False)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--bias", action="store_true", default=False)
    parser.add_argument("--gridsearch", action="store_true", default=False)

    parser.add_argument("--gmm", action="store_true", default=False)
    parser.add_argument("--knn", action="store_true", default=False)
    parser.add_argument("--parzen", action="store_true", default=False)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = arg_parsing()

    dataset_params = {
        "n_samples": 100,
        "seed": 10009,
        "target_type": "GMM",
        "validation_size": 0,
    }
    mlp_params = {
        "dropout": 0.000,
        "hidden_layer": [(64, nn.Tanh()), (56, nn.Tanh()), (38, nn.Tanh())],
        "last_activation": None,  # None or lambda
    }

    train_params = {
        "epochs": 540,
        "batch_size": 26,
        "loss_type": "mse_loss",  # "huber_loss" or "mse_loss"
        "optimizer": "Adam",  # "RMSprop" or "Adam"
        "learning_rate": 0.000874345,
    }

    gmm_target_params = {
        "n_components": 10,
        "n_init": 100,
        "max_iter": 80,
        "init_params": "k-means++",  # "k-means++" or "random" or "kmeans" or "random_from_data"
        "random_state": 10009,
    }

    device = "cpu"

    set_seed(dataset_params["seed"])

    # choose the pdf for the experiment
    if args.pdf in ["exponential", "exp"]:
        pdf = PDF(default="EXPONENTIAL_06")
    else:
        pdf = PDF(default="MULTIVARIATE_1254")

    # load dataset
    pdf.generate_training(n_samples=dataset_params["n_samples"])
    pdf.generate_validation(n_samples=dataset_params["validation_size"])
    pdf.generate_test()

    gm_model = GaussianMixture(**gmm_target_params)
    _, target_y = gen_target_with_gm_parallel(gm_model, X=pdf.training_X, n_jobs=-1, progress_bar=True)

    model = NeuralNetworkModular(
        dropout=mlp_params["dropout"],
        hidden_layer=mlp_params["hidden_layer"],
        last_activation=mlp_params["last_activation"],
    )

    train_loss, val_loss, _, _ = training(
        model,
        pdf.training_X,
        target_y,
        pdf.validation_X,
        pdf.validation_Y,
        train_params["learning_rate"],
        train_params["epochs"],
        train_params["batch_size"],
        train_params["optimizer"],
        train_params["loss_type"],
        device,
    )

    # evaluate model
    metrics = evaluation(model, pdf.test_X, pdf.test_Y, device)

    print(metrics["r2"])
    exit()

    # experiement 1:
    # - cambiare il nunmero di componenti GMM, tenendo fermo la rete MLP e vedere comme cambia l'r2 e kl

    metrics_changed_1 = []
    start_components = gmm_target_params["n_components"]
    components = range(1, 8, 2)

    for c in components:

        print("generating target with GMM")
        gmm_target_params["n_components"] = c
        gm_model = GaussianMixture(**gmm_target_params)
        _, target_y = gen_target_with_gm_parallel(gm_model, X=pdf.training_X, n_jobs=-1, progress_bar=True)

        model = NeuralNetworkModular(
            dropout=mlp_params["dropout"],
            hidden_layer=mlp_params["hidden_layer"],
            last_activation=mlp_params["last_activation"],
        )

        train_loss, val_loss, _, _ = training(
            model,
            pdf.training_X,
            target_y,
            pdf.validation_X,
            pdf.validation_Y,
            train_params["learning_rate"],
            train_params["epochs"],
            train_params["batch_size"],
            train_params["optimizer"],
            train_params["loss_type"],
            device,
        )

        # evaluate model
        metrics = evaluation(model, pdf.test_X, pdf.test_Y, device)

        metrics_changed_1.append(metrics)

    r2_values = list(map(operator.itemgetter("r2"), metrics_changed_1))
    kl_values = list(map(operator.itemgetter("kl"), metrics_changed_1))

    sns.set_style("darkgrid")
    plt.plot(list(components), r2_values, marker="o", label="R2 score")
    plt.plot(list(components), kl_values, marker="o", label="KL score")

    # Aggiungere la legenda
    plt.legend(
        title=f"Changing Components, Best {pdf.name} with {dataset_params['n_samples']}", labels=["R2", "KL"]
    )

    # Etichettare gli assi
    plt.xlabel("Number of Components")
    plt.ylabel("Score")

    # Mostrare il grafico
    plt.show()

    # experiment 2:
    # - cambiare il numero di neuroni del primo layer della mlp vedere come cambia l'r2:
    metrics_changed_2 = []
    neurons = range(2, 64, 2)
    gmm_target_params["n_components"] = start_components
    gm_model = GaussianMixture(**gmm_target_params)
    _, target_y = gen_target_with_gm_parallel(gm_model, X=pdf.training_X, n_jobs=-1, progress_bar=True)

    for neuron in neurons:

        mlp_params["hidden_layer"][0][0] = neuron

        model = NeuralNetworkModular(
            dropout=mlp_params["dropout"],
            hidden_layer=mlp_params["hidden_layer"],
            last_activation=mlp_params["last_activation"],
        )

        train_loss, val_loss, _, _ = training(
            model,
            pdf.training_X,
            target_y,
            pdf.validation_X,
            pdf.validation_Y,
            train_params["learning_rate"],
            train_params["epochs"],
            train_params["batch_size"],
            train_params["optimizer"],
            train_params["loss_type"],
            device,
        )

        metrics = evaluation(model, pdf.test_X, pdf.test_Y, device)

        metrics_changed_2.append(metrics)

    r2_values = list(map(operator.itemgetter("r2"), metrics_changed_2))
    kl_values = list(map(operator.itemgetter("kl"), metrics_changed_2))

    sns.set_style("darkgrid")
    plt.plot(list(neurons), r2_values, marker="o", label="R2 score")
    plt.plot(list(neurons), kl_values, marker="o", label="KL score")

    # Aggiungere la legenda
    plt.legend(
        title=f"Changing Neurons, Best {pdf.name} with {dataset_params['n_samples']}", labels=["R2", "KL"]
    )

    # Etichettare gli assi
    plt.xlabel("Number of Neurons")
    plt.ylabel("Score")

    # Mostrare il grafico
    plt.show()
