import numpy as np
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import r2_score, mean_squared_error, max_error, explained_variance_score
from sklearn.mixture import GaussianMixture

from scipy.stats import entropy
import re
import argparse

# ----
from utils.data_manager import PDF
from utils.utils import plot_AllInOne, write_result, generate_unique_id
from model.nn_model import GM_NN_Model
from model.gm_model import GaussianMixtureModel


def calculate_kl_divergence(true_pdf, predicted_pdf):
    kl_divergence = entropy(true_pdf, predicted_pdf)
    return np.mean(kl_divergence)


def calculate_ise(true_pdf, predicted_pdf, bin_width=0.01):
    # Calcola le aree dei rettangoli tra le due distribuzioni
    rectangle_areas = (true_pdf - predicted_pdf) ** 2 * bin_width
    ise = np.sum(rectangle_areas)
    return ise


def test_and_log(
    model,
    X,
    y_true,
    pdf: list = [],
    n_samples: int = 100,
    n_components=4,
    mlp_params="None",
    best_params="None",
    model_type="GMM",
    dimension: int = 1,
    id=000,
    write_to_csv=True,
):
    pattern = re.compile(r"MLP|NN")
    if not bool(pattern.search(model_type)):
        # predict the pdf with GMM
        pdf_predicted = np.exp(
            model.score_samples(X)
        )  # with .score_samples we get the log-likelihood over all the samples

    else:
        # predict the pdf with GMM + MLP
        pdf_predicted = model.predict(X.astype(np.float32))

    if write_to_csv == True:
        # MLP scoring:
        write_result(
            id=id,
            pdf_type=pdf,
            components=n_components,
            model_type=model_type,
            experiment_params=mlp_params,
            best_params=best_params,
            n_samples=n_samples,
            dimension=dimension,
            r2_score=r2_score(y_true, pdf_predicted),
            mse_score=np.sqrt(mean_squared_error(y_true, pdf_predicted)),
            max_error_score=max_error(y_true, pdf_predicted),
            evs_score=explained_variance_score(y_true, pdf_predicted),
            ise_score=calculate_ise(y_true, pdf_predicted),
            k1_score=calculate_kl_divergence(y_true, pdf_predicted),
        )

    return pdf_predicted


def main():
    parser = argparse.ArgumentParser(description="Project for AI Exam")
    parser.add_argument("--rate", type=int, default=0.6)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--bias", action="store_true", default=False)

    args = parser.parse_args()

    seed = 42

    # Parameters:
    n_components = args.components  # number of components for the Gaussian Mixture
    rate = args.rate  # rate of the exponential distribution PDF
    n_samples = args.samples  # number of samples to generate from the exp distribution
    limit_test = (0, 20)  # range limits for the x-axis of the test set
    stepper_x_test = 0.01  # step to take on the limit_test for generate the test data
    init_param_gmm = "random"  # the initialization of the mean vector for the base GMM
    init_param_mlp = "random"  # the initialization of the mean vector for the GMM in the GMM+MLP model
    max_iter = 100  # the maximum number of iterations for training the GMMs
    n_init = 10  # the number of initial iterations for training the GMMs

    # # # best params:
    # mlp_params = {
    #     "criterion": [nn.MSELoss],
    #     "max_epochs": [50],
    #     "batch_size": [4],
    #     "lr": [0.005],
    #     "module__n_layer": [2],
    #     "module__last_activation": ["lambda"],
    #     "module__num_units": [80],
    #     "module__activation": [nn.ReLU()],
    #     "module__type_layer": ["increase"],
    #     "optimizer": [optim.Adam],
    #     "module__dropout": [0.3],
    # }

    mlp_params = {
        "criterion": [nn.HuberLoss],
        "max_epochs": [500, 800],
        "batch_size": [16, 32],
        "lr": [0.001, 0.005],
        "module__last_activation": ["lambda"],
        "module__hidden_layer": [
            [(64, nn.ReLU()), (64, nn.ReLU())],
            [(16, nn.Tanh()), (32, nn.Tanh()), (64, nn.ReLU())],
            [(32, nn.ReLU()), (64, nn.Tanh()), (32, nn.ReLU())],
            [(32, nn.ReLU()), (16, nn.Tanh()), (8, nn.ReLU())],
        ],
        "optimizer": [optim.Adam],
        "optimizer__weight_decay": [0.001, 0],
        "module__dropout": [0.2, 0.5],
    }

    id = generate_unique_id(
        [init_param_mlp, init_param_gmm, n_init, mlp_params, seed, n_components, n_samples, rate], lenght=3
    )

    # generate the sample from a exponential distribution:
    # pdf = PDF({"type": "exponential", "mean": rate})
    pdf = PDF(
        [
            {"type": "logistic", "mean": 20, "scale": 0.5, "weight": 0.4},
            {"type": "logistic", "mean": 10, "scale": 4, "weight": 0.4},
            {"type": "logistic", "mean": 17, "scale": 1, "weight": 0.2},
        ]
    )

    x_training, y_training = pdf.generate_training(n_samples=n_samples, save_filename=f"train.npy", seed=seed)
    offset_limit = 0
    limit_test = (np.min(x_training) - offset_limit, np.max(x_training) + offset_limit)
    # generate the data for plotting the pdf
    x_test, y_test = pdf.generate_test(
        save_filename=f"test.npy",
        range_limit=limit_test,
        stepper=stepper_x_test,
    )

    # ------------------------ GMM: --------------------------
    # train the model
    model_gmm = GaussianMixtureModel(n_components=n_components, seed=seed, init_params=init_param_gmm)
    model_gmm.fit(x_training)

    # predict the pdf with GMM
    pdf_predicted_gmm = test_and_log(
        model_gmm,
        X=x_test,
        y_true=y_test,
        pdf=pdf.params,
        dimension=pdf.dimension,
        n_samples=n_samples,
        n_components=n_components,
        model_type=f"GMM {init_param_gmm}",
        id=id,
        write_to_csv=True,
    )

    # ------------------------ GMM + NN: --------------------------

    model_mlp = GM_NN_Model(
        parameters=mlp_params, n_components=n_components, bias=args.bias, init_params=init_param_mlp, seed=seed
    )

    # make the model
    if args.gpu == True:
        device = "cuda"
    else:
        device = "cpu"

    _, y_mlp = model_mlp.fit(
        x_training,
        search_type="auto",
        n_jobs=args.jobs,
        device=device,
        save_filename=f"train_mlp{'_Biased' if args.bias == True else '' }_{init_param_mlp}_C{n_components}.npy",
    )

    # train the model and predict the pdf over the test set
    pdf_predicted_mlp = test_and_log(
        model_mlp,
        X=x_test,
        y_true=y_test,
        pdf=pdf.params,
        dimension=pdf.dimension,
        n_samples=n_samples,
        n_components=n_components,
        mlp_params=mlp_params,
        best_params=model_mlp.nn_model.best_params_,
        id=id,
        model_type=f"MLP {init_param_mlp} {'Biased' if args.bias == True else '' }",
        write_to_csv=True,
    )

    # plot the real pdf and the predicted pdf for GMM and MLP
    plot_AllInOne(
        x_training,
        x_test,
        pdf_predicted_mlp=pdf_predicted_mlp,
        pdf_predicted_gmm=pdf_predicted_gmm,
        pdf_true=y_test,
        save=False,
        name=f"result-{id}_C{n_components}",
        title=f"PDF estimation with {n_components} components",
        show=args.show,
    )


if __name__ == "__main__":
    main()
