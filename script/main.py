import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error, max_error, explained_variance_score
from scipy.stats import entropy
import re

import argparse


# ----
from data_manager import (
    load_test,
    load_training,
    load_training_MLP_Label,
)
from utils import plot_AllInOne, write_result, generate_unique_id
from model.nn_model import NerualNetwork_model
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
    y,
    rate=1.0,
    n_samples=100,
    n_components=4,
    mlp_params="None",
    best_params="None",
    model_type="GMM",
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
            rate=rate,
            experiment_type=n_components,
            model_type=model_type,
            experiment_params=mlp_params,
            best_params=best_params,
            n_samples=n_samples,
            r2_score=r2_score(y, pdf_predicted),
            mse_score=np.sqrt(mean_squared_error(y, pdf_predicted)),
            max_error_score=max_error(y, pdf_predicted),
            evs_score=explained_variance_score(y, pdf_predicted),
            ise_score=calculate_ise(y, pdf_predicted),
            k1_score=calculate_kl_divergence(y, pdf_predicted),
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
    limit_test = (0, 10)  # range limits for the x-axis of the test set
    stepper_x_test = 0.001  # step to take on the limit_test for generate the test data
    init_param_gmm = "random"  # the initialization of the mean vector for the base GMM
    init_param_mlp = "random"  # the initialization of the mean vector for the GMM in the GMM+MLP model
    max_iter = 100  # the maximum number of iterations for training the GMMs
    n_init = 10  # the number of initial iterations for training the GMMs

    mlp_params = {
        "criterion": [nn.MSELoss],
        "max_epochs": [50, 80],
        "batch_size": [4, 8, 16],
        "lr": [0.005, 0.01, 0.015],
        "module__n_layer": [2, 3],
        "module__last_activation": ["lambda", nn.ReLU()],
        "module__num_units": [80, 50, 10],
        "module__activation": [nn.ReLU(), nn.Tanh()],
        "module__type_layer": ["increase", "decrease"],
        "optimizer": [optim.Adam],
        "module__dropout": [0.3, 0.5, 0.0],
    }
    

    id = generate_unique_id(
        [init_param_mlp, init_param_gmm, n_init, mlp_params, seed, n_components, n_samples, rate], lenght=3
    )

    # generate the sample from a exponential distribution:
    x_training, _ = load_training(f"training_N{n_samples}_R{rate}.npy", n_samples=n_samples, rate=rate, seed=seed)

    # generate the data for plotting the pdf
    x_test, y_test = load_test(
        f"test_L({limit_test[0]}-{limit_test[1]})_S{stepper_x_test}.npy",
        rate=rate,
        range_limit=limit_test,
        stepper=stepper_x_test,
    )

    # ------------------------ GMM: --------------------------
    # train the model
    model_gmm = GaussianMixtureModel(
        n_components=n_components, seed=seed, n_init=n_init, max_iter=max_iter, init_params=init_param_gmm
    )
    model_gmm.fit(x_training)

    # predict the pdf with GMM
    pdf_predicted_gmm = test_and_log(
        model_gmm,
        X=x_test,
        y=y_test,
        rate=rate,
        n_samples=n_samples,
        n_components=n_components,
        model_type=f"GMM {init_param_gmm}",
        id=id,
        write_to_csv=True,
    )

    # ------------------------ GMM + NN: --------------------------
    # generate MLP label from GMM unbiased
    _, y_mlp = load_training_MLP_Label(
        x_training,
        n_components=n_components,
        init_params=init_param_mlp,
        load_file=f"training_mlp{'_Biased' if args.bias == True else '' }_{init_param_mlp}_C{n_components}_R{rate}.npy",
        seed=seed,
        bias=args.bias,
    )

    # make the model
    if args.gpu == True:
        device = "cuda"
    else:
        device = "cpu"
        
    model_mlp = NerualNetwork_model(parameters=mlp_params, search="gridsearch", device=device, n_jobs=args.jobs)

    # train the model and predict the pdf over the test set
    model_mlp.fit(x_training.astype(np.float32), y_mlp.astype(np.float32))

    print(model_mlp.best_params_)

    pdf_predicted_mlp = test_and_log(
        model_mlp,
        X=x_test,
        y=y_test,
        rate=rate,
        n_samples=n_samples,
        n_components=n_components,
        mlp_params=mlp_params,
        best_params=model_mlp.best_params_,
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
        save=True,
        name=f"result-{id}_C{n_components}_R{rate}",
        title=f"PDF estimation with {n_components} components",
        show=args.show,
    )


if __name__ == "__main__":
    main()
