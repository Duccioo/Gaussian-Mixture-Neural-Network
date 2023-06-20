import numpy as np

# ----
from data_manager import generate_training, generate_test, save_dataset, load_dataset
from utils import plot_AllInOne, plot_histo, plot_2pdf
from model.nn_model import NerualNetwork_model
from model.gm_model import GaussianMixtureModel_bias


def main():
    seed = 42

    # generate the sample from a exponential distribution:
    rate = 1
    x_training, y_training = generate_training(rate=rate, seed=seed)

    # generate the data for plotting the pdf
    x_test, y_test = generate_test(rate=rate, limit_x=10)

    # train the model
    model = GaussianMixtureModel_bias(n_components=8, seed=seed)
    model.fit(x_training)

    pdf_predicted = np.exp(model.score_samples(x_test))

    # plot the real pdf and the predicted pdf
    # plot_2pdf(x_test, y_test, x_test, pdf_predicted)
    plot_AllInOne(x_training, x_test, pdf_predicted=pdf_predicted, pdf_true=y_test)

    # plot_histo(x_training)


if __name__ == "__main__":
    main()
