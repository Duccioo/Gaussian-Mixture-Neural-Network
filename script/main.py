import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error, max_error, explained_variance_score
import scipy.stats as stats


# ----
from data_manager import (
    generate_training,
    generate_test,
    save_dataset,
    load_dataset,
    generate_training_MLP_Label,
    load_training_MLP_Label,
)
from utils import plot_AllInOne, plot_histo, plot_2pdf, write_result
from model.nn_model import NerualNetwork_model
from model.gm_model import GaussianMixtureModel_bias


def calculate_kl_divergence(true_pdf, predicted_pdf):
    kl_divergence = stats.entropy(true_pdf, predicted_pdf)
    return kl_divergence


def calculate_ise(true_pdf, predicted_pdf, bin_width=0.01):
    # Calcola le aree dei rettangoli tra le due distribuzioni
    rectangle_areas = (true_pdf - predicted_pdf) ** 2 * bin_width
    ise = np.sum(rectangle_areas)
    return ise


def main():
    seed = 42

    n_components = 8
    # generate the sample from a exponential distribution:
    rate = 0.6
    x_training, y_training = generate_training(rate=rate, seed=seed)

    # generate the data for plotting the pdf
    x_test, y_test = generate_test(rate=rate, limit_x=10)

    # train the model
    model_gmm = GaussianMixtureModel_bias(n_components=n_components, seed=seed)
    model_gmm.fit(x_training)
    # predict the pdf with GMM
    pdf_predicted_gmm = np.exp(model_gmm.score_samples(x_test))

    # generate MLP label from GMM unbiased
    x_training, y_mlp = load_training_MLP_Label(
        x_training, n_components=n_components, load_file=f"training_mlp_{n_components}.npy"
    )

    # make the model
    params = {
        "criterion": [nn.MSELoss],
        "max_epochs": [50, 100],
        "batch_size": [16],
        "lr": [0.01],
        "module__n_layer": [1, 2],
        "module__last_activation": ["lambda"],
        "module__num_units": [100, 50],
        "module__activation": [nn.Tanh(), nn.Sigmoid(), nn.LeakyReLU(0.6), nn.ReLU()],
    }

    model_mlp = NerualNetwork_model(
        parameters=params,
        optimizer=optim.Adam,
        module__dropout=0.5,
        module__type_layer="increase",
        search="gridsearch",
    )

    # train the model and predict the pdf over the test set
    model_mlp.fit(x_training.astype(np.float32), y_mlp.astype(np.float32))
    print(model_mlp.best_params_)
    pdf_predicted_mlp = model_mlp.predict(x_test.astype(np.float32))

    # scoring:
    print("R2 SCORE (best 1.0)")
    print("MLP: ", r2_score(y_test, pdf_predicted_mlp))
    print("GMM: ", r2_score(y_test, pdf_predicted_gmm))

    print("\nMSE score (less is better)")
    print("MLP: ", np.sqrt(mean_squared_error(y_test, pdf_predicted_mlp)))
    print("GMM: ", np.sqrt(mean_squared_error(y_test, pdf_predicted_gmm)))

    print("\nMax Error (less is better)")
    print("MLP: ", max_error(y_test, pdf_predicted_mlp))
    print("GMM: ", max_error(y_test, pdf_predicted_gmm))

    print("\nExplained Variance Score: (best 1.0)")
    print("MLP: ", explained_variance_score(y_test, pdf_predicted_mlp))
    print("GMM: ", explained_variance_score(y_test, pdf_predicted_gmm))

    print("\nISE (lower is better)")
    print("MLP: ", calculate_ise(y_test, pdf_predicted_mlp))
    print("GMM: ", calculate_ise(y_test, pdf_predicted_gmm))

    print("\nK1 divergence (lower is better)")
    print("MLP: ", calculate_kl_divergence(y_test, pdf_predicted_mlp))

    write_result(
        experiment_type=n_components,
        experiment_params=params,
        best_params=model_mlp.best_params_,
        r2_score=r2_score(y_test, pdf_predicted_mlp),
        mse_score=np.sqrt(mean_squared_error(y_test, pdf_predicted_mlp)),
        max_error_score=max_error(y_test, pdf_predicted_mlp),
        evs_score=explained_variance_score(y_test, pdf_predicted_mlp),
        ise_score=calculate_ise(y_test, pdf_predicted_mlp),
        k1_score=calculate_kl_divergence(y_test, pdf_predicted_mlp),
    )

    # plot the real pdf and the predicted pdf for GMM and MLP
    plot_AllInOne(
        x_training, x_test, pdf_predicted_mlp=pdf_predicted_mlp, pdf_predicted_gmm=pdf_predicted_gmm, pdf_true=y_test
    )


if __name__ == "__main__":
    main()
