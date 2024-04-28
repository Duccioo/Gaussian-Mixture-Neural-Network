import torch
import torch.nn as nn

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture
from torchsummary import summary

# ---
from utils.data_manager import PDF
from model.nn_model import NeuralNetworkModular
from model.gm_model import gen_target_with_gm_parallel
from utils.utils import set_seed


def train_old_style(
    model,
    X_train,
    Y_train,
    lr,
    epochs,
    batch_size,
    optimizer_name,
    device,
    criterion=nn.HuberLoss(),
):

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)

    xy_train = torch.cat((X_train, Y_train), 1)

    train_loader = torch.utils.data.DataLoader(
        xy_train,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    # Training of the model.
    for epoch in range(epochs):
        model.train()
        for batch_idx, train_data in enumerate(train_loader):
            data = train_data[:, 0]
            target = train_data[:, 1]
            # Limiting training data for faster epochs.

            data, target = data.view(data.size(0), 1).to(device), target.view(
                target.size(0), 1
            ).to(device)

            optimizer.zero_grad()
            output = model(data)

            loss_value = criterion(output, target)
            loss_value.backward()
            optimizer.step()


if __name__ == "__main__":

    # load dataset
    pdf = PDF(default="MULTIVARIATE_1254")
    bias = False
    stepper_x_test = 0.01
    n_samples = 50

    seed = 8
    n_components = 4
    init_params_gmm = "k-means++"
    n_init = 70
    max_iter = 30

    # set seed
    set_seed(seed)
    # torch.manual_seed(seed)

    # parametri della rete neurale:
    epochs = 770
    batch_size = 34
    lr = 0.0007619622536581125
    optimizer_name = "RMSprop"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load parameters
    dropout = 0.00
    hidden_layer = [
        (50, nn.ReLU()),
        (50, nn.Tanh()),
        (22, nn.Tanh()),
        (44, nn.Sigmoid()),
    ]
    last_activation = "lambda"

    pdf.generate_training(n_samples=n_samples, seed=seed)

    # generate the data for plotting the pdf
    X_test, Y_test = pdf.generate_test(stepper=stepper_x_test)

    gm_model = GaussianMixture(
        n_components=n_components,
        init_params=init_params_gmm,
        random_state=27,
        n_init=n_init,
        max_iter=max_iter,
    )

    _, Y_predicted = gen_target_with_gm_parallel(
        gm_model, X=pdf.training_X, n_jobs=-1, progress_bar=True
    )

    model = NeuralNetworkModular(
        dropout=dropout,
        hidden_layer=hidden_layer,
        last_activation=last_activation,
    )

    # print(model)

    print(summary(model, verbose=0))

    train_old_style(
        model,
        pdf.training_X,
        Y_predicted,
        lr,
        epochs,
        batch_size,
        optimizer_name,
        device,
        nn.MSELoss(),
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
    sns.scatterplot(
        x=pdf.training_X.flatten(), y=Y_predicted.flatten(), color="purple", label="GMM"
    )

    plt.legend()
    plt.show()
