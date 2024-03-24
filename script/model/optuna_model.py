import optuna

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture


import os

# ---
from model.nn_model import AdaptiveSigmoid
from utils.data_manager import PDF
from model.gm_model import gen_target_with_gm_parallel
from model.parzen_model import ParzenWindow_Model, gen_target_with_parzen_parallel
from utils.utils import generate_unique_id, set_seed


def define_MLP(trial: optuna.Trial, params: dict = {}):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int(
        "n_layers", params["n_layers"][0], params["n_layers"][1]
    )
    layers = []

    in_features = 1
    for i in range(n_layers):
        out_features = trial.suggest_int(
            "n_units_l{}".format(i),
            params["range_neurons"][0],
            params["range_neurons"][1],
        )

        out_attivation = trial.suggest_categorical(
            "activation_l{}".format(i), params["suggest_activation"]
        )

        layers.append(nn.Linear(in_features, out_features))
        nn.init.xavier_normal_(layers[-1].weight)

        if out_attivation == "relu":
            layers.append(nn.ReLU())
        elif out_attivation == "tanh":
            layers.append(nn.Tanh())
        elif out_attivation == "sigmoid":
            layers.append(nn.Sigmoid())
        try:
            if isinstance(params["range_dropout"], (list, tuple)):
                p = trial.suggest_float(
                    "dropout_l{}".format(i),
                    params["range_dropout"][0],
                    params["range_dropout"][1],
                    log=True,
                )
            else:
                p = params["range_dropout"]

            layers.append(nn.Dropout(p))

        except:
            pass

        in_features = out_features

    layers.append(nn.Linear(in_features, 1))
    nn.init.xavier_normal_(layers[-1].weight)

    last_activation = trial.suggest_categorical(
        "last_activation", params["suggest_last_activation"]
    )

    if last_activation == "lambda":
        layers.append(AdaptiveSigmoid())
    elif last_activation == "sigmoid":
        layers.append(nn.Sigmoid())
    elif last_activation == "relu":
        layers.append(nn.ReLU())

    model = nn.Sequential(*layers)

    return model


def objective_MLP(
    trial: optuna.Trial, X_train, Y_train, X_test, Y_test, params, device
):

    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)

    # Generate the model.
    model = define_MLP(trial, params=params).to(device)

    # Generate the optimizers.
    lr = trial.suggest_float(
        "learning_rate",
        params["learning_rate"][0],
        params["learning_rate"][1],
        log=True,
    )

    # loss_name = trial.suggest_categorical("loss", params["loss"])
    s_epoch = trial.suggest_int("epoch", params["epoch"][0], params["epoch"][1])
    optimizer_name = trial.suggest_categorical("optimizer", params["optimizer"])
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    batch_size = trial.suggest_int(
        "batch_size", params["batch_size"][0], params["batch_size"][1]
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)

    xy_train = torch.cat((X_train, Y_train), 1)

    train_loader = torch.utils.data.DataLoader(
        xy_train,
        batch_size=batch_size,
        shuffle=True,
    )

    # Training of the model.
    for epoch in range(s_epoch):
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

            loss_value = F.huber_loss(output, target)
            loss_value.backward()
            optimizer.step()

        # Validation of the model each epoch.
        model.eval()
        with torch.no_grad():
            output = model(X_test)

        r2_value = r2_score(output.numpy(), Y_test)
        # mse = mean_squared_error(output.numpy(), Y_test)

        trial.report(r2_value, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return r2_value


def objective_MLP_allin_gmm(trial: optuna.Trial, params, tmp_dir, device):
    """
    A function to optimize the parameters of a multilayer perceptron (MLP) using Gaussian Mixture Model (GMM) for the given dataset parameters.

    Args:
        trial (optuna.Trial): An Optuna trial object used to manage the optimization process.
        params (dict): A dictionary containing the parameters for the dataset and MLP optimization.
            It should contain the following keys:
                for the dataset parameters:
                - 'seed': tuple of two integers representing the range of seed values.
                - 'n_init': tuple of two integers representing the range of n_init values.
                - 'max_iter': tuple of two integers representing the range of max_iter values.
                - 'n_components': tuple of two integers representing the range of n_components values.
                - 'n_samples': tuple of two integers representing the range of n_samples values.
                - 'init_params_gmm': list of strings representing the initialization parameters choices (e.g., 'kmeans', 'random').

                for the model params:
                - 'batch_size': tuple of two integers representing the range of batch_size values.
                - 'optimizer': list of strings representing the optimizer choices (e.g., 'Adam', 'RMSprop', 'SGD').
                - 'learning_rate': tuple of two floats representing the range of learning_rate values.
                - 'loss': list of strings representing the loss function choices (e.g., 'mse', 'mae', 'huber').
                - 'epoch': tuple of two integers representing the range of epoch values.
                - 'range_dropout': tuple of two floats representing the range of dropout values.
                - 'n_layers': tuple of two integers representing the range of n_layers values.
                - 'range_neurons': tuple of two integers representing the range of neurons values.
                - 'suggest_activation': list of strings representing the activation function choices (e.g., 'tanh', 'sigmoid', 'relu').
                - 'suggest_last_activation': list of strings representing the last activation function choices (e.g., 'lambda', 'sigmoid', 'relu').

    Returns:
        float: The R-squared score obtained after optimizing the MLP with GMM for the given dataset parameters.
    """

    # è un wrapper del objective_MLP per scegliere anche i parametri del dataset

    pdf = PDF(default="MULTIVARIATE_1254")
    bias = False
    stepper_x_test = 0.01  # genero con la multivariate circa 1000 esempi

    if isinstance(params["seed"], (list, tuple)):
        seed = trial.suggest_int("seed", params["seed"][0], params["seed"][1])
    else:
        seed = params["seed"]

    set_seed(seed)

    if isinstance(params["n_samples"], (list, tuple)):
        n_samples = trial.suggest_int(
            "n_samples", params["n_samples"][0], params["n_samples"][1]
        )
    else:
        n_samples = params["n_samples"]

    x_training, _ = pdf.generate_training(n_samples=n_samples, seed=seed)

    # generate the data for plotting the pdf
    x_test, y_test = pdf.generate_test(stepper=stepper_x_test)

    if params["target_type"] == "GMM":

        if isinstance(params["n_init"], (list, tuple)):
            n_init = trial.suggest_int(
                "n_init", params["n_init"][0], params["n_init"][1], step=10
            )
        else:
            n_init = params["n_init"]

        if isinstance(params["max_iter"], (list, tuple)):
            max_iter = trial.suggest_int(
                "max_iter", params["max_iter"][0], params["max_iter"][1], step=10
            )
        else:
            max_iter = params["max_iter"]

        if isinstance(params["n_components"], (list, tuple)):
            n_components = trial.suggest_int(
                "n_components", params["n_components"][0], params["n_components"][1]
            )
        else:
            n_components = params["n_components"]

        if isinstance(params["init_params_gmm"], (list, tuple)):
            init_params_gmm = trial.suggest_categorical(
                "init_params_gmm", params["init_params_gmm"]
            )
        else:
            init_params_gmm = params["init_params_gmm"]

        gm_model = GaussianMixture(
            n_components=n_components,
            init_params=init_params_gmm,
            random_state=seed,
            n_init=n_init,
            max_iter=max_iter,
        )

        # generate the id
        unique_id_gmm_target = generate_unique_id(
            [x_training, n_components, bias, init_params_gmm, seed], 5
        )
        file_name = f"target_gm_C{n_components}_S{n_samples}_P{init_params_gmm}_N{n_init}_M{max_iter}.npz"
        file_path = os.path.join(tmp_dir, file_name)

        _, gen_target_y = gen_target_with_gm_parallel(
            gm_model=gm_model,
            X=x_training,
            save_filename=file_path,
            progress_bar=True,
            n_jobs=-1,
        )

    elif params["target_type"] == "PARZEN":
        if isinstance(params["h"], (list, tuple)):
            h = trial.suggest_float("h", params["h"][0], params["h"][1])
        else:
            h = params["h"]

        file_name = f"target_parzen_S{n_samples}_H{h}.npz"
        file_path = os.path.join(tmp_dir, file_name)
        parzen_model = ParzenWindow_Model(h=h)

        _, gen_target_y = gen_target_with_parzen_parallel(
            parzen_model,
            X=x_training,
            save_filename=file_path,
            progress_bar=True,
            n_jobs=-1,
        )

    r2_score = objective_MLP(
        trial, x_training, gen_target_y, x_test, y_test, params, device
    )

    return r2_score
