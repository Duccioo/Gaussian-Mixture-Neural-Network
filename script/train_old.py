import torch
import torch.nn as nn

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture


# ---
from utils.data_manager import PDF
from model.nn_model import train_old_style, NeuralNetworkModular
from model.gm_model import gen_target_with_gm_parallel


if __name__ == "__main__":

    # load dataset
    pdf = PDF(default="MULTIVARIATE_1254")
    bias = False
    stepper_x_test = 0.01
    n_samples = 100
    seed = 42
    n_components = 11
    init_params_gmm = "kmeans"
    n_init = 60
    max_iter = 80

    # set seed
    torch.manual_seed(seed)

    # parametri della rete neurale:
    epochs = 658
    batch_size = 27
    lr = 0.0010039468848053604
    optimizer_name = "RMSprop"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load parameters
    dropout = 0.01
    hidden_layer = [(52, nn.ReLU()), (10, nn.Tanh()), (36, nn.ReLU()), (21, nn.ReLU())]
    last_activation = "lambda"

    X_train, Y_train = pdf.generate_training(n_samples=n_samples, seed=seed)

    # generate the data for plotting the pdf
    X_test, Y_test = pdf.generate_test(stepper=stepper_x_test)

    gm_model = GaussianMixture(
        n_components=n_components,
        init_params=init_params_gmm,
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
    )

    _, Y_predicted = gen_target_with_gm_parallel(
        gm_model, X=X_train, n_jobs=3, progress_bar=True, save_filename="prova-ddd.npz"
    )

    model = NeuralNetworkModular(
        dropout=dropout,
        hidden_layer=hidden_layer,
        last_activation=last_activation,
        device=device,
    )

    train_old_style(
        model, X_train, Y_predicted, lr, epochs, batch_size, optimizer_name, device
    )

    # evaluate model
    model.eval()
    with torch.no_grad():
        y_predicted_mlp = model(torch.tensor(X_test, dtype=torch.float32))
        y_predicted_mlp = y_predicted_mlp.detach().numpy()
        r2 = r2_score(Y_test, y_predicted_mlp)
        print(r2)

    # print figure

    sns.lineplot(x=X_test.flatten(), y=Y_test.flatten(), color="green", label="True")
    sns.lineplot(
        x=X_test.flatten(), y=y_predicted_mlp.flatten(), label="base", color="red"
    )
    sns.scatterplot(x=X_train.flatten(), y=Y_predicted.flatten(), color="purple", label="GMM")

    plt.legend()
    plt.show()
