import numpy as np
import numba as nb
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    max_error,
    explained_variance_score,
)
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy
import argparse

# ----
from utils.data_manager import PDF
from utils.utils import plot_AllInOne, write_result, generate_unique_id, check_base_dir
from model.nn_model import GM_NN_Model
from model.parzen_model import ParzenWindow_Model
from model.knn_model import KNN_Model
from model.gm_model import gen_target_with_gm_parallel


def calculate_kl_divergence(true_pdf, predicted_pdf):
    kl_divergence = entropy(true_pdf, predicted_pdf)
    return np.mean(kl_divergence)


@nb.njit()
def calculate_ise(true_pdf, predicted_pdf, bin_width=0.01):
    # Calcola le aree dei rettangoli tra le due distribuzioni
    rectangle_areas = (true_pdf - predicted_pdf) ** 2 * bin_width
    ise = np.sum(rectangle_areas)
    return ise


def get_model_complexity(parameters):
    nn_param = 1
    nn_layer = 0
    for neuron in parameters["module__hidden_layer"]:
        nn_param = nn_param * neuron[0]
        nn_layer = nn_layer + 1
    return nn_param, nn_layer


def test_and_log(
    y_true,
    y_predicted,
    pdf_type: str = "None",
    pdf: list = [],
    n_samples: int = 100,
    n_components=None,
    n_layer: int = None,
    n_neurons: int = None,
    mlp_params="None",
    best_params="None",
    model_type="GMM",
    epoch: int = None,
    dimension: int = 1,
    id: str = "",
    id_dataset: str = "",
    id_experiment: str = "",
    write_to_csv=True,
):
    round_number = 3
    r2_value = round(r2_score(y_true, y_predicted), round_number)

    if write_to_csv == True:
        # MLP scoring:
        write_result(
            id=str(id),
            id_experiment=id_experiment,
            pdf_type=pdf_type,
            components=n_components,
            model_type=model_type,
            experiment_params=mlp_params,
            best_params=best_params,
            n_samples=n_samples,
            dimension=dimension,
            r2_score=r2_value,
            max_error_score=round(max_error(y_true, y_predicted), round_number),
            n_layer=n_layer,
            n_neurons=n_neurons,
            mse_score=round(
                np.sqrt(mean_squared_error(y_true, y_predicted)), round_number
            ),
            evs_score=round(
                explained_variance_score(y_true, y_predicted), round_number
            ),
            ise_score=round(calculate_ise(y_true, y_predicted), round_number),
            k1_score=round(calculate_kl_divergence(y_true, y_predicted), round_number),
            epoch=epoch,
            pdf_param=pdf,
            id_dataset=id_dataset,
        )

    return r2_value


def main():
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

    # select device:
    if args.gpu:
        device = args.gpu
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parameters:
    # -- data parameters
    seed = 42
    n_samples = args.samples  # number of samples to generate from the exp distribution
    stepper_x_test = 0.01  # step to take on the limit_test for generate the test data

    # -- gmm parameters
    n_components = args.components  # number of components for the Gaussian Mixture
    max_iter = 80  # the maximum number of iterations for training the GMMs
    n_init = 60  # the number of initial iterations for training the GMMs

    # -- other models parameters
    init_param_gmm = "kmeans"  # the initialization of the mean vector for the base GMM [random, kmeans, k-means++, random_from_data]
    parzen_h = 0.33
    knn_k1 = 2

    # -- mlp parameters
    init_param_mlp = "kmeans"  # the initialization of the mean vector for the GMM in the GMM+MLP model [random, kmeans, k-means++, random_from_data]
    early_stop = None  # "valid_loss" or "r2" or None
    patience = 20
    mlp_params = {
        "criterion": [nn.HuberLoss()],
        "max_epochs": [658],
        "batch_size": [37, 32],
        "lr": [
            0.0010039468848053604,
        ],
        "module__last_activation": ["lambda"],
        "module__hidden_layer": [
            [(52, nn.ReLU()), (10, nn.Tanh()), (36, nn.ReLU()), (21, nn.ReLU())],
            [(60, nn.ReLU()), (60, nn.ReLU()), (10, nn.ReLU())],
            [(54, nn.ReLU()), (57, nn.ReLU())],
            [(32, nn.ReLU()), (16, nn.Tanh()), (16, nn.Tanh()), (8, nn.Tanh())],
            # [(16, nn.ReLU()), (32, nn.Tanh()), (32, nn.Tanh()), (16, nn.ReLU())],
            # [(128, nn.ReLU()), (32, nn.Tanh())],
            # [(16, nn.ReLU()), (128, nn.ReLU())],
            # [(128, nn.ReLU()), (128, nn.ReLU()), (128, nn.ReLU())],
            # [(32, nn.LeakyReLU()), (32, nn.Tanh()), (64, nn.ReLU())],
            # [(128, nn.ReLU()), (64, nn.Tanh()), (32, nn.ReLU())],
        ],
        "optimizer": [optim.RMSprop],
        # "optimizer__weight_decay": [0.001],
        "module__dropout": [0.01],
    }

    # generate the sample from a known distribution:
    pdf_exponential = PDF(
        {"type": "exponential", "mean": 0.6}, name="exponential standard"
    )

    pdf_logistic_multimodal = PDF(
        [
            [
                {"type": "logistic", "mean": 20, "scale": 0.5, "weight": 0.4},
                {"type": "logistic", "mean": 10, "scale": 4, "weight": 0.4},
                {"type": "logistic", "mean": 17, "scale": 1, "weight": 0.2},
            ],
        ],
        name="multimodal 3 logistic",
    )

    if args.pdf in ["exponential", "exp"]:
        pdf = pdf_exponential
    elif args.pdf in ["multimodal logistic", "logistic"]:
        pdf = pdf_logistic_multimodal
    else:
        pdf = PDF(default="MULTIVARIATE_1254")

    # sample the data from a known distribution
    x_training, y_training = pdf.generate_training(n_samples=n_samples, seed=seed)
    n_samples = x_training.shape[0]

    # generate the data for plotting the pdf
    x_test, y_test = pdf.generate_test(stepper=stepper_x_test)

    id_dataset = generate_unique_id([x_training, y_training, x_test, y_test], lenght=5)
    id_experiment = generate_unique_id(
        [
            x_training,
            y_training,
            x_test,
            y_test,
            args,
            mlp_params,
            seed,
            n_samples,
            pdf,
            stepper_x_test,
            n_components,
            init_param_mlp,
            init_param_gmm,
            parzen_h,
            knn_k1,
            early_stop,
            patience,
        ],
        lenght=7,
    )

    # ------------------------ GMM: --------------------------
    if args.gmm is True:
        id_gmm = generate_unique_id(
            [
                x_training,
                x_test,
                y_test,
                seed,
                n_components,
                n_samples,
                init_param_gmm,
                max_iter,
                n_init,
            ],
            lenght=5,
        )

        # train the GMM model
        model_gmm = GaussianMixture(
            n_components=n_components,
            random_state=seed,
            init_params=init_param_gmm,
            max_iter=max_iter,
            n_init=n_init,
        )
        model_gmm.fit(x_training)
        # predict the pdf with GMM
        pdf_predicted_gmm = np.exp(model_gmm.score_samples(x_test))
        # with .score_samples we get the log-likelihood over all the samples

        # predict the pdf with GMM
        r2_gmm = test_and_log(
            y_true=y_test,
            y_predicted=pdf_predicted_gmm,
            pdf_type=pdf.name,
            pdf=pdf.params,
            dimension=pdf.dimension,
            n_samples=n_samples,
            n_components=n_components,
            model_type=f"GMM {init_param_gmm}",
            id=id_gmm,
            id_dataset=id_dataset,
            id_experiment=id_experiment,
            write_to_csv=args.save,
        )
    else:
        pdf_predicted_gmm = None

    # ------------------------ PARZEN WINDOW: --------------------------
    if args.parzen:
        id_parzen = generate_unique_id(
            [x_training, x_test, y_test, seed, n_samples, parzen_h], lenght=5
        )
        model_parzen = ParzenWindow_Model(h=parzen_h)
        model_parzen.fit(training=x_training)
        pdf_predicted_parzen = model_parzen.predict(test=x_test)
        r2_parzen = test_and_log(
            y_true=y_test,
            y_predicted=pdf_predicted_parzen,
            pdf_type=pdf.name,
            pdf=pdf.params,
            dimension=pdf.dimension,
            n_samples=n_samples,
            model_type=f"PARZEN WINDOW",
            id=id_parzen,
            id_dataset=id_dataset,
            id_experiment=id_experiment,
            write_to_csv=args.save,
            mlp_params=parzen_h,
        )
    else:
        pdf_predicted_parzen = None

    # ------------------------ KNN: --------------------------
    if args.knn:
        id_knn = generate_unique_id(
            [x_training, x_test, y_test, seed, n_samples, knn_k1], lenght=5
        )
        model_knn = KNN_Model(k1=knn_k1)
        model_knn.fit(training=x_training)
        pdf_predicted_knn = model_knn.predict(test=x_test)
        r2_knn = test_and_log(
            y_true=y_test,
            y_predicted=pdf_predicted_knn,
            pdf_type=pdf.name,
            pdf=pdf.params,
            dimension=pdf.dimension,
            n_samples=n_samples,
            model_type=f"KNN",
            id=id_knn,
            id_dataset=id_dataset,
            id_experiment=id_experiment,
            write_to_csv=args.save,
            mlp_params=knn_k1,
        )
    else:
        pdf_predicted_knn = None

    # ------------------------ GMM + NN: --------------------------
    id_mlp = generate_unique_id(
        [
            x_training,
            x_test,
            y_test,
            seed,
            n_components,
            n_samples,
            init_param_mlp,
            mlp_params,
            args.bias,
            args.gridsearch,
            early_stop,
            patience,
        ],
        lenght=5,
    )

    gm_model_target = GaussianMixture(
        n_components=n_components,
        init_params=init_param_mlp,
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
    )

    model_mlp = GM_NN_Model(
        parameters=mlp_params,
        n_components=n_components,
        bias=args.bias,
        init_params=init_param_mlp,
        seed=seed,
    )

    # generate the id
    unique_id = generate_unique_id(
        [x_training, n_components, args.bias, init_param_mlp, seed], 5
    )

    # check if a saved target file exists:

    base_dir = ["..", "data", "MLP"]
    base_dir = check_base_dir(base_dir)

    save_filename = f"train_mlp{'_Biased' if args.bias == True else '' }_{init_param_mlp}_C{n_components}"

    if save_filename is not None:
        save_filename = save_filename.split(".")[0]
        save_filename = save_filename + "_" + unique_id + ".npz"
        save_filename = os.path.join(base_dir, save_filename)

    _, gmm_target_y = gen_target_with_gm_parallel(
        gm_model=gm_model_target,
        X=x_training,
        save_filename=save_filename,
        bias=args.bias,
        progress_bar=True,
        n_jobs=3,
    )

    model_mlp.fit(
        x_training,
        gmm_target_y,
        search_type="auto" if args.gridsearch == True else None,
        n_jobs=args.jobs,
        device=device,
        early_stop=early_stop,
        patience=patience,
    )

    # predict the pdf with GMM + MLP
    pdf_predicted_mlp = model_mlp.predict(x_test.astype(np.float32))

    if args.gridsearch == True:
        epoch = model_mlp.nn_best_params["max_epochs"]
    else:
        epoch = model_mlp.nn_model.history[-1, "epoch"]

    num_neurons, num_layer = get_model_complexity(model_mlp.nn_best_params)

    # train the model and predict the pdf over the test set
    r2_mlp = test_and_log(
        y_predicted=pdf_predicted_mlp,
        y_true=y_test,
        pdf_type=pdf.name,
        pdf=pdf.params,
        dimension=pdf.dimension,
        n_samples=n_samples,
        n_components=n_components,
        n_layer=num_layer,
        n_neurons=num_neurons,
        mlp_params=mlp_params,
        best_params=model_mlp.nn_best_params,
        id=id_mlp,
        epoch=epoch,
        id_dataset=id_dataset,
        model_type=f"MLP {init_param_mlp}{' Biased' if args.bias == True else '' }",
        id_experiment=id_experiment,
        write_to_csv=args.save,
    )

    # with open("prova.json", "w") as f:
    #     json.dump(model_mlp.history, f)

    # r2_mlp_history = [elem["r2"] for elem in model_mlp.history]
    # train_loss_mlp_history = [elem["train_loss"] for elem in model_mlp.history]
    # valid_loss_mlp_history = [elem["valid_loss"] for elem in model_mlp.history]

    # ----------------------------------------------------------------
    # plot the real pdf and the predicted pdf for GMM and MLP
    if pdf.dimension == 1:
        plot_AllInOne(
            x_training,
            x_test,
            mlp_target=gmm_target_y if args.mlp_targets else None,
            pdf_predicted_knn=pdf_predicted_knn,
            pdf_predicted_parzen=pdf_predicted_parzen,
            pdf_predicted_mlp=pdf_predicted_mlp,
            pdf_predicted_gmm=pdf_predicted_gmm,
            pdf_true=y_test,
            save=args.save,
            name=f"result_{id_experiment}",
            title=f"PDF estimation with {n_components} components and {n_samples} samples",
            show=args.show,
        )
    else:
        print(f"impossible to plot on a {pdf.dimension} dimensional space")

    print(id_experiment)
    print("MLP R2: ", r2_mlp)


if __name__ == "__main__":
    main()
