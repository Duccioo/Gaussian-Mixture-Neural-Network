import argparse
import operator
import numpy as np


import torch
import torch.nn as nn

from sklearn.mixture import GaussianMixture

import seaborn as sns
import matplotlib.pyplot as plt

from rich.progress import track

# ---
from utils.data_manager import PDF
from model.nn_model import NeuralNetworkModular
from model.gm_model import gen_target_with_gm_parallel
from utils.utils import set_seed
from script.training_MLP import training, evaluation


def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)


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
        "n_samples": args.samples,
        "seed": 98,
        "target_type": "GMM",
        "validation_size": 0,
    }
    mlp_params = {
        "dropout": 0.000,
        "hidden_layer": [
            [26, nn.Sigmoid()],
            (26, nn.Tanh()),
            (24, nn.Tanh()),
            (54, nn.ReLU()),
        ],
        "last_activation": None,  # None or lambda
    }

    train_params = {
        "epochs": 360,
        "batch_size": 40,
        "loss_type": "mse_loss",  # "huber_loss" or "mse_loss"
        "optimizer": "Adam",  # "RMSprop" or "Adam"
        "learning_rate": 0.00266,
    }

    gmm_target_params = {
        "n_components": 7,
        "n_init": 10,
        "max_iter": 30,
        "init_params": "k-means++",  # "k-means++" or "random" or "kmeans" or "random_from_data"
        "random_state": 62,
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

    torch.save(model.state_dict(), "model_inizialization.pth")

    training(
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
    metrics, _ = evaluation(model, pdf.test_X, pdf.test_Y, device)

    print(
        f"R2 di partenza con {gmm_target_params['n_components']} e {mlp_params['hidden_layer'][0][0]}",
        metrics["r2"],
    )

    # experiment 1:
    # - change the number of GMM components, keeping the MLP network fixed and see how it changes the R2 and KL

    metrics_changed_1 = []
    start_components = gmm_target_params["n_components"]
    components = range(1, 30)

    progress_bar = track(components, description="Experiment 1")

    for c in progress_bar:

        print("generating target with GMM")
        gmm_target_params["n_components"] = c
        gm_model = GaussianMixture(**gmm_target_params)
        _, target_y = gen_target_with_gm_parallel(gm_model, X=pdf.training_X, n_jobs=-1, progress_bar=False)

        set_seed(dataset_params["seed"])

        model_1 = NeuralNetworkModular(
            dropout=mlp_params["dropout"],
            hidden_layer=mlp_params["hidden_layer"],
            last_activation=mlp_params["last_activation"],
        )

        # weight reset
        model_1.load_state_dict(torch.load("model_inizialization.pth"))

        training(
            model_1,
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
        metrics, _ = evaluation(model_1, pdf.test_X, pdf.test_Y, device)

        metrics_changed_1.append(metrics)
        print(f"{c} --> r2", metrics["r2"], "kl", metrics["kl"])

    print("-----")

    r2_values = list(map(operator.itemgetter("r2"), metrics_changed_1))
    kl_values = list(map(operator.itemgetter("kl"), metrics_changed_1))

    # kl_values = [1 if x > 2 else x for x in kl_values]

    print(
        "------- BEST R2 ::: ",
        np.max(r2_values),
        f"CON {components[np.argmax(r2_values)]} numero di componenti",
    )

    f1 = plt.figure()

    sns.set_style("darkgrid")
    plt.plot(list(components), r2_values, marker="o", label="R2 score")
    plt.plot(list(components), kl_values, marker="o", label="KL score")

    # Aggiungere la legenda
    plt.legend(
        labels=["R2 score", "KL score"],
    )

    # Etichettare gli assi
    plt.xlabel("Number of Components")
    plt.ylabel("Score")
    plt.ylim(-0.5, 2)

    # Mostrare il grafico
    plt.savefig(f"changing_components_best_{pdf.name}_{dataset_params['n_samples']}.png")
    plt.show()
    plt.close(f1)

    # experiment 2:
    # - cambiare il numero di neuroni del primo layer della mlp vedere come cambia l'r2:
    metrics_changed_2 = []
    neurons = range(1, 200)
    gmm_target_params["n_components"] = start_components
    gm_model = GaussianMixture(**gmm_target_params)
    _, target_y = gen_target_with_gm_parallel(gm_model, X=pdf.training_X, n_jobs=-1, progress_bar=False)

    progress_bar = track(neurons, description="Experiment 2")

    for neuron in progress_bar:

        mlp_params["hidden_layer"][0][0] = neuron

        # print(mlp_params["hidden_layer"])

        set_seed(dataset_params["seed"])

        model = NeuralNetworkModular(
            dropout=mlp_params["dropout"],
            hidden_layer=mlp_params["hidden_layer"],
            last_activation=mlp_params["last_activation"],
        )

        # Reset dei pesi
        model_1.load_state_dict(torch.load("model_inizialization.pth"))

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

        metrics, _ = evaluation(model, pdf.test_X, pdf.test_Y, device)

        metrics_changed_2.append(metrics)
        print(neuron, " --> r2", metrics["r2"], "kl", metrics["kl"])

    r2_values = list(map(operator.itemgetter("r2"), metrics_changed_2))
    kl_values = list(map(operator.itemgetter("kl"), metrics_changed_2))
    # kl_values = [1 if x > 2 else x for x in kl_values]

    print(
        "------- Best R2 ::: ",
        np.max(r2_values),
        f"CON {neurons[np.argmax(r2_values)]} numero di neuroni",
    )

    f2 = plt.figure()

    sns.set_style("darkgrid")
    plt.plot(list(neurons), r2_values, marker="o", label="R2 score")
    plt.plot(list(neurons), kl_values, marker="o", label="KL score")
    plt.ylim(-0.5, 2)

    # Aggiungere la legenda
    plt.legend(
        labels=["R2 score", "KL score"],
    )

    # Etichettare gli assi
    plt.xlabel("Number of Neurons")
    plt.ylabel("Score")

    # Mostrare il grafico
    plt.savefig(f"changing_neurons_best_{pdf.name}_{dataset_params['n_samples']}.png")
    plt.show()
