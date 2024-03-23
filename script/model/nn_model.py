import torch.nn as nn
import torch
import torch.nn.init as init
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from skorch import NeuralNet
from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler, WandbLogger
import numpy as np
from attrs import define, field
import os

import random

# ---
from utils.utils import check_base_dir, generate_unique_id, set_seed

BASE_DATA_DIR = ["..", "..", "data", "MLP"]


class AdaptiveSigmoid(nn.Module):
    def __init__(self, lambda_init=1.0):
        super(AdaptiveSigmoid, self).__init__()
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lambda_param * self.sigmoid(x)
        return x


class NeuralNetworkModular(nn.Module):
    def __init__(
        self,
        input_features: int = 1,
        output_features: int = 1,
        dropout: int = 0.5,
        hidden_layer: list = [(10, nn.ReLU())],
        last_activation=False,
        device="cpu",
    ):
        super(NeuralNetworkModular, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.activation = nn.ModuleList()
        self.batchNorm = nn.ModuleList()

        self.layers.append(nn.Linear(input_features, hidden_layer[0][0]).to(device))
        self.activation.append(hidden_layer[0][1])

        for i in range(len(hidden_layer) - 1):
            self.layers.append(
                nn.Linear(hidden_layer[i][0], hidden_layer[i + 1][0]).to(device)
            )
            self.activation.append(hidden_layer[i + 1][1])

        self.output_layer = nn.Linear(hidden_layer[-1][0], output_features).to(device)

        if last_activation == "lambda":
            self.last_activation = AdaptiveSigmoid()
        else:
            self.last_activation = last_activation

        for layer in self.layers:
            init.xavier_normal_(layer.weight)
        init.xavier_normal_(self.output_layer.weight)

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activation):
            x = self.dropout(activation(layer(x)))
        x = self.output_layer(x)

        if self.last_activation is not None:
            x = self.last_activation(x)
        return x


@define(slots=True)
class GM_NN_Model:
    parameters: dict = field(factory=dict)
    criterion: list = field(factory=list, init=True)
    n_components: int = field(default=4, init=True)
    bias: bool = field(default=False, init=True)
    init_params: str = field(default="random", init=True)
    base_dir: str = field(init=True, default=check_base_dir(BASE_DATA_DIR))
    seed: int = field(default=42, init=True)

    nn_model: NeuralNet = NeuralNet(NeuralNetworkModular, nn.MSELoss)
    nn_best_params: dict = field(factory=dict)
    gmm_target_x: np.ndarray = field(init=True, default=np.array(None))
    gmm_target_y: np.ndarray = field(init=True, default=np.array(None))
    history: dict = field(factory=dict)

    def __init__(
        self,
        parameters: dict = {},
        n_components: int = 4,
        bias: bool = False,
        init_params: str = "random",
        base_dir: list = None,
        seed: int = None,
    ):
        if base_dir is None:
            base_dir = check_base_dir(BASE_DATA_DIR)
        else:
            base_dir = check_base_dir(base_dir)

        if (
            isinstance(parameters, list) == False
            and parameters.get("criterion") == None
        ):
            raise ValueError("Please specify a valid criterion!")
        criterion = parameters.get("criterion")

        if isinstance(criterion, list) == False:
            criterion = [criterion]

        self.__attrs_init__(
            parameters,
            criterion=criterion,
            n_components=n_components,
            bias=bias,
            init_params=init_params,
            base_dir=base_dir,
            seed=seed,
        )

    def __attrs_post_init__(self):
        if self.seed is not None:
            
            set_seed(self.seed)
            # torch.manual_seed(self.seed)
        # setup the Gaussian Mixture Model:

    def fit(
        self,
        X,
        Y: np.ndarray,
        search_type: str = None,
        n_jobs: int = 1,
        device: str = "cpu",
        patience: int = 20,
        early_stop: bool = False,
    ):
        if device == "auto" or device == None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        callbacks = []
        # callbacks.append(
        #     EpochScoring(scoring="r2", lower_is_better=False),
        # )
        # callbacks.append(LRScheduler(policy=torch.optim.lr_scheduler.LinearLR))

        # wandb_run = wandb.init()

        if early_stop != False and early_stop in ["valid_loss", "r2"]:

            callbacks.append(
                EarlyStopping(
                    monitor=early_stop,
                    patience=patience,
                    load_best=False,
                    lower_is_better=False,
                ),
            )

        if search_type == "auto":
            for value in self.parameters.values():
                if isinstance(value, list):
                    search_type = "gridsearch"
                    break

        if search_type == None:
            new_dict = {}
            for key, value in self.parameters.items():
                if isinstance(value, list) and len(value) > 0:
                    new_dict[key] = value[0]
                else:
                    new_dict[key] = value

            # wandb_run.config.update(new_dict)
            # callbacks.append(WandbLogger(wandb_run))

            self.nn_model = NeuralNet(
                NeuralNetworkModular,
                **new_dict,
                verbose=1,
                device=device,
                module__device=device,
                module__input_features=1,
                callbacks=callbacks,
            )
            

        elif search_type == "gridsearch":
            new_dict = {}
            for key, value in self.parameters.items():
                if not isinstance(value, list):
                    new_dict[key] = [value]
                else:
                    new_dict[key] = value

            nn_model_net = NeuralNet(
                NeuralNetworkModular,
                self.criterion[0],
                verbose=0,
                device=device,
                module__device=device,
                module__input_features=X.shape[1],
                callbacks=callbacks,
            )

            self.nn_model = GridSearchCV(
                nn_model_net,
                new_dict,
                refit="r2",
                cv=5,
                verbose=3,
                n_jobs=n_jobs,
                scoring=["r2"],
            )

        self.nn_model.fit(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32),
        )
        # print("HISTORY", self.nn_model.history)

        # print(self.nn_model.history)

        if search_type == "gridsearch":
            self.nn_best_params = self.nn_model.best_params_
            print(self.nn_model.cv_results_)

        else:
            self.nn_best_params = new_dict
            self.history = self.nn_model.history

        return self.nn_model, self.gmm_target_y

    def predict(self, X):
        return self.nn_model.predict(torch.tensor(X, dtype=torch.float32))


def train_old_style(
    model, X_train, Y_train, lr, epochs, batch_size, optimizer_name, device
):

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)

    xy_train = torch.cat((X_train, Y_train), 1)

    train_loader = torch.utils.data.DataLoader(
        xy_train,
        batch_size=batch_size,
        shuffle=True,
    )

    criterion = nn.HuberLoss()
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
    num_samples = 1000
    test_size = 0.2
    random_state = 42
    rate = 0.5

    # Generate random number for a random dataset
    np.random.seed(random_state)
    x = np.random.exponential(scale=1 / rate, size=num_samples)
    y = rate * np.exp(x * (-rate))

    # split the dataset into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    x_train = x_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # create the model
    params = {
        "criterion": nn.MSELoss,
        "max_epochs": [100, 50],
        "module__num_units": [80, 150],
        "module__last_activation": "lambda",
    }

    model = GM_NN_Model(
        parameters=params,
        n_components=4,
        bias=False,
        init_params="random",
        seed=random_state,
    )
    m = model.fit(x_train, n_jobs=4, save_filename="prova.npy")

    # test the model
    out = m.predict(torch.Tensor(x_test))
    print(out[0:5].reshape(-1).round(3))
    print(y_test[0:5].reshape(-1).round(3))
