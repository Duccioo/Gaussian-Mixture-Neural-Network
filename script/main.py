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
    pdf_type: str = "None",
    pdf: list = [],
    n_samples: int = 100,
    n_components=4,
    mlp_params="None",
    best_params="None",
    model_type="GMM",
    epoch: int = 0,
    dimension: int = 1,
    id: str = "",
    id_dataset: str = "",
    write_to_csv=True,
):
    pattern = re.compile(r"MLP|NN")
    if not bool(pattern.search(model_type)):
        # predict the pdf with GMM
        pdf_predicted = np.exp(
            model.score_samples(X)
        )  # with .score_samples we get the log-likelihood over all the samples
        # print(X.shape)
        # print(pdf_predicted[0:10])

    else:
        # predict the pdf with GMM + MLP
        pdf_predicted = model.predict(X.astype(np.float32))

    round_number = 3
    r2_value = round(r2_score(y_true, pdf_predicted), round_number)

    if write_to_csv == True:
        # MLP scoring:
        write_result(
            id=str(id),
            pdf_type=pdf_type,
            components=n_components,
            model_type=model_type,
            experiment_params=mlp_params,
            best_params=best_params,
            n_samples=n_samples,
            dimension=dimension,
            r2_score=r2_value,
            mse_score=round(np.sqrt(mean_squared_error(y_true, pdf_predicted)), round_number),
            max_error_score=round(max_error(y_true, pdf_predicted), round_number),
            evs_score=round(explained_variance_score(y_true, pdf_predicted), round_number),
            ise_score=round(calculate_ise(y_true, pdf_predicted), round_number),
            k1_score=round(calculate_kl_divergence(y_true, pdf_predicted), round_number),
            epoch=epoch,
            pdf_param=pdf,
            id_dataset=id_dataset,
        )

    return (pdf_predicted, r2_value)


def main():
    # command line parsing
    parser = argparse.ArgumentParser(description="Project for AI Exam")
    parser.add_argument("--pdf", type=str, default="exponential")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--bias", action="store_true", default=False)
    parser.add_argument("--gridsearch", action="store_true", default=False)
    args = parser.parse_args()

    # Parameters:
    seed = 42
    n_components = args.components  # number of components for the Gaussian Mixture
    n_samples = args.samples  # number of samples to generate from the exp distribution
    stepper_x_test = 0.01  # step to take on the limit_test for generate the test data
    init_param_gmm = "random"  # the initialization of the mean vector for the base GMM [random, kmeans, k-means++, random_from_data]
    init_param_mlp = "kmeans"  # the initialization of the mean vector for the GMM in the GMM+MLP model [random, kmeans, k-means++, random_from_data]
    max_iter = 100  # the maximum number of iterations for training the GMMs
    n_init = 10  # the number of initial iterations for training the GMMs
    offset_limit = 0.0

    mlp_params = {
        "criterion": [nn.HuberLoss],
        "max_epochs": [60, 40, 50],
        "batch_size": [4, 8],
        "lr": [0.001, 0.0015, 0.002, 0.005],
        "module__last_activation": ["lambda"],
        "module__hidden_layer": [
            [(64, nn.ReLU())],
            [(16, nn.ReLU()), (32, nn.ReLU()), (32, nn.ReLU()), (16, nn.ReLU())],
            [(128, nn.ReLU()), (128, nn.Tanh())],
            [(80, nn.ReLU()), (160, nn.ReLU())],
            [(128, nn.ReLU()), (128, nn.ReLU()), (128, nn.ReLU())],
            [(64, nn.Tanh()), (32, nn.Tanh()), (128, nn.ReLU())],
            [(32, nn.LeakyReLU()), (32, nn.Tanh()), (64, nn.ReLU())],
            [(128, nn.ReLU()), (64, nn.Tanh()), (32, nn.ReLU())],
        ],
        "optimizer": [optim.Adam],
        # "optimizer__weight_decay": [0.001],
        "module__dropout": [0.3, 0.1],
    }

    # generate the sample from a known distribution:
    pdf_exponential = PDF({"type": "exponential", "mean": 0.6})

    pdf_multimodal = PDF(
        [
            [
                {"type": "logistic", "mean": 20, "scale": 0.5, "weight": 0.4},
                {"type": "logistic", "mean": 10, "scale": 4, "weight": 0.4},
                {"type": "logistic", "mean": 30, "scale": 1, "weight": 0.2},
            ],
        ]
    )

    # pdf_multimodal.generate_training(100, 112311)
    # pdf_multimodal.generate_test((0,10), 0.1)

    # pdf_multimodal.test_X
    # pdf_multimodal.test_Y
    # pdf_multimodal.training_X
    # pdf_multimodal.training_Y

    if args.pdf in ["exponential", "exp"]:
        pdf = pdf_exponential
    else:
        pdf = pdf_multimodal

    # sample the data from a known distribution
    x_training, y_training = pdf.generate_training(n_samples=n_samples, save_filename=f"train", seed=seed)

    # generate the data for plotting the pdf
    limit_test = (np.min(x_training) - offset_limit, np.max(x_training) + offset_limit)
    x_test, y_test = pdf.generate_test(
        save_filename=f"test",
        range_limit=limit_test,
        stepper=stepper_x_test,
    )

    id_dataset = generate_unique_id([x_training, y_training, x_test, y_test], lenght=5)

    # ------------------------ GMM: --------------------------
    id_gmm = generate_unique_id(
        [x_training, x_test, y_test, seed, n_components, n_samples, init_param_gmm, max_iter, n_init], lenght=5
    )

    # train the GMM model
    model_gmm = GaussianMixture(
        n_components=n_components, random_state=seed, init_params=init_param_gmm, max_iter=max_iter, n_init=n_init
    )
    model_gmm.fit(x_training)

    # predict the pdf with GMM
    pdf_predicted_gmm, r2_gmm = test_and_log(
        model_gmm,
        X=x_test,
        y_true=y_test,
        pdf_type=args.pdf,
        pdf=pdf.params,
        dimension=pdf.dimension,
        n_samples=n_samples,
        n_components=n_components,
        model_type=f"GMM {init_param_gmm}",
        id=id_gmm,
        id_dataset=id_dataset,
        write_to_csv=True,
    )

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
        ],
        lenght=5,
    )
    model_mlp = GM_NN_Model(
        parameters=mlp_params, n_components=n_components, bias=args.bias, init_params=init_param_mlp, seed=seed
    )

    # make the model
    if args.gpu == True:
        device = "cuda"
    else:
        device = "cpu"

    model_mlp.fit(
        x_training,
        search_type="auto" if args.gridsearch == True else None,
        n_jobs=args.jobs,
        device=device,
        save_filename=f"train_mlp{'_Biased' if args.bias == True else '' }_{init_param_mlp}_C{n_components}",
    )

    if args.gridsearch == True:
        epoch = model_mlp.nn_best_params["max_epochs"]
    else:
        epoch = model_mlp.nn_model.history[-1, "epoch"]

    # train the model and predict the pdf over the test set
    pdf_predicted_mlp, r2_mlp = test_and_log(
        model_mlp,
        X=x_test,
        y_true=y_test,
        pdf_type=args.pdf,
        pdf=pdf.params,
        dimension=pdf.dimension,
        n_samples=n_samples,
        n_components=n_components,
        mlp_params=mlp_params,
        best_params=model_mlp.nn_best_params,
        id=id_mlp,
        epoch=epoch,
        id_dataset=id_dataset,
        model_type=f"MLP {init_param_mlp} {'Biased' if args.bias == True else '' }",
        write_to_csv=True,
    )

    # plot the real pdf and the predicted pdf for GMM and MLP
    if pdf.dimension == 1:
        plot_AllInOne(
            x_training,
            x_test,
            mlp_target=model_mlp.gmm_target_y,
            pdf_predicted_mlp=pdf_predicted_mlp,
            pdf_predicted_gmm=pdf_predicted_gmm,
            pdf_true=y_test,
            save=args.save,
            name=f"result_G-{id_gmm}_M-{id_mlp}_C{n_components}",
            title=f"PDF estimation with {n_components} components",
            show=args.show,
        )
    else:
        print(f"impossible to plot on a {pdf.dimension} dimensional space")

    print(f"id for gmm {id_gmm} || id for mlp: {id_mlp}")
    print(f"R2 GMM -> {r2_gmm} \nR2 MLP -> {r2_mlp}")


if __name__ == "__main__":
    main()
