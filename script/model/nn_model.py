import torch.nn as nn
import torch
import torch.nn.init as init
from skorch import NeuralNet, NeuralNetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
from attrs import define, field
from sklearn.mixture import GaussianMixture
import os
from itertools import product
from tqdm import tqdm
from skorch.callbacks import Callback

# ---
from utils.utils import check_base_dir, generate_unique_id
from model.gm_model import generate_target_MLP

BASE_DATA_DIR = ["..", "..", "data_2", "MLP"]

# pbar = tqdm(total=120, desc="Training progress")


# Callback personalizzata per la stampa della progress bar
class ProgressBarCallback(Callback):
    def __init__(self, pbar=None) -> None:
        super().__init__()
        self.pbar = pbar

    def on_train_end(self, net, X, y):
        self.pbar.update(1)  # Aggiorna la barra di progressione
        self.pbar.set_postfix({"Training": self.pbar.n})  # Aggiorna il messaggio nella barra


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
        self.layers = []
        self.activation = []

        # check if neurons are all paired with an activation function:
        # if len(num_units) != len(activation):
        #     raise ValueError("The number of units not match the number of activation functions")

        self.layers.append(nn.Linear(input_features, hidden_layer[0][0], device=device))
        self.activation.append(hidden_layer[0][1])

        for i, (neurons, activation) in enumerate(hidden_layer):
            if i != len(hidden_layer) - 1:
                self.layers.append(nn.Linear(int(neurons), int(hidden_layer[i + 1][0]), device=device))
                self.activation.append(activation)

        self.layers = nn.ModuleList(self.layers)
        # print(self.layers)
        self.output_layer = nn.Linear(int(hidden_layer[-1][0]), output_features, device=device)

        for layer in self.layers:
            init.xavier_normal_(layer.weight)

        if last_activation == "lambda":
            self.last_activation = AdaptiveSigmoid()
        else:
            self.last_activation = last_activation

        init.xavier_normal_(self.output_layer.weight)

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activation):
            x = activation(layer(x))
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

    gm_model: GaussianMixture = field(factory=GaussianMixture)
    nn_model: NeuralNet = NeuralNet(NeuralNetworkModular, nn.MSELoss)

    def __init__(
        self,
        parameters: dict = {},
        n_components: int = 4,
        bias: bool = False,
        init_params: str = "random",
        base_dir: list or None = None,
        seed: int or None = None,
    ):
        if base_dir is None:
            base_dir = check_base_dir(BASE_DATA_DIR)
        else:
            base_dir = check_base_dir(base_dir)

        if isinstance(parameters, list) == False and parameters.get("criterion") == None:
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
            torch.manual_seed(self.seed)

        # setup the Gaussian Mixture Model:
        self.gm_model = GaussianMixture(self.n_components, init_params=self.init_params, random_state=self.seed)

    def fit(
        self,
        X,
        search_type: str or None = None,
        n_jobs: int = 1,
        device: str or None = "cpu",
        save_filename: str or None = None,
        base_dir: str or None = None,
    ):
        if device == "auto" or device == None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

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

            self.nn_model = NeuralNet(
                NeuralNetworkModular,
                **new_dict,
                verbose=1,
                device=device,
                module__device=device,
                module__input_features=X.shape[1],
            )

        elif search_type == "gridsearch":
            new_dict = {}
            for key, value in self.parameters.items():
                if not isinstance(value, list):
                    new_dict[key] = [value]
                else:
                    new_dict[key] = value

            total_train_count = len(list(product(*self.parameters.values()))) * 5
            progress_bar = tqdm(total=total_train_count, desc="Training", disable=False, leave=True)

            nn_model_net = NeuralNet(
                NeuralNetworkModular,
                self.criterion[0],
                verbose=0,
                device=device,
                module__device=device,
                module__input_features=X.shape[1],
                # callbacks=[ProgressBarCallback(pbar=progress_bar)],
            )

            self.nn_model = GridSearchCV(
                nn_model_net,
                new_dict,
                refit="r2",
                cv=5,
                verbose=0,
                n_jobs=n_jobs,
                # pre_dispatch=2,
                scoring=["r2", "max_error", "explained_variance"],
            )

        # generate the id
        unique_id = generate_unique_id([X, self.n_components, self.bias, self.init_params, self.seed], 5)

        # check if a saved target file exists:
        if base_dir is None:
            base_dir = self.base_dir

        if save_filename is not None:
            save_filename = save_filename.split(".")[0]
            save_filename = save_filename + "_" + unique_id + ".npy"
            save_filename = os.path.join(base_dir, save_filename)

        _, Y = generate_target_MLP(gm_model=self.gm_model, X=X, save_filename=save_filename, bias=self.bias)

        self.nn_model.fit(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
        # plt.scatter(X, Y)
        # plt.show()

        # print(r2_score(Y, self.nn_model.predict(X.astype(np.float32))))
        # print("datii", Y[0:10], X[0:10])

        return self.nn_model, Y

    def predict(self, X):
        return self.nn_model.predict(torch.tensor(X, dtype=torch.float32))


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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
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

    model = GM_NN_Model(parameters=params, n_components=4, bias=False, init_params="random", seed=random_state)
    m = model.fit(x_train, n_jobs=4, save_filename="prova.npy")

    # test the model
    out = m.predict(torch.Tensor(x_test))
    print(out[0:5].reshape(-1).round(3))
    print(y_test[0:5].reshape(-1).round(3))
