import torch.nn as nn
import torch
import torch.nn.init as init

# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# from skorch import NeuralNet
# from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler, WandbLogger
import numpy as np
from attrs import define, field
import os

import random

# ---
# from utils.utils import check_base_dir, generate_unique_id, set_seed

BASE_DATA_DIR = ["..", "..", "data", "MLP"]


class AdaptiveSigmoid(nn.Module):
    def __init__(self, lambda_init=1.0):
        super(AdaptiveSigmoid, self).__init__()
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lambda_param * self.sigmoid(x)
        return x


def mlp_block(input, output, activation="relu", *args, **kwargs):
    activations = nn.ModuleDict([["tanh", nn.Tanh()], ["relu", nn.ReLU()]])

    return nn.Sequential(
        nn.Linear(input, output, *args, **kwargs),
        activations[activation],
    )


def last_block(input, output, last_activarion):
    activations = nn.ModuleDict([["lambda", AdaptiveSigmoid()], [None, None]])

    return nn.Sequential(
        nn.Linear(
            input,
            output,
        ),
        activations[last_activarion],
    )


class NeuralNetworkModular(nn.Module):
    def __init__(
        self,
        input_features: int = 1,
        output_features: int = 1,
        dropout: int = 0.5,
        hidden_layer: list = [(10, nn.ReLU())],
        last_activation=False,
    ):
        """
        Initializes a new instance of the NeuralNetworkModular class.

        Parameters:
            input_features (int, optional): The number of input features. Defaults to 1.
            output_features (int, optional): The number of output features. Defaults to 1.
            dropout (float, optional): The dropout rate. Defaults to 0.5.
            hidden_layer (list, optional): A list of tuples representing the hidden layers. Each tuple contains the number of neurons and the activation function. Defaults to [(10, nn.ReLU())].
            last_activation (bool or callable, optional): The activation function for the last layer. If set to "lambda", an AdaptiveSigmoid activation function is used. If set to None, no activation function is used. Defaults to False.

        Returns:
            None
        """
        super(NeuralNetworkModular, self).__init__()
        self.dropout = nn.Dropout(dropout)
        layers = []
        activation = []

        layer = nn.Linear(input_features, hidden_layer[0][0])
        init.xavier_normal_(layer.weight)
        layers.append(layer)
        activation.append(hidden_layer[0][1])

        for i in range(len(hidden_layer) - 1):
            layer = nn.Linear(hidden_layer[i][0], hidden_layer[i + 1][0])
            init.xavier_normal_(layer.weight)
            layers.append(layer)
            activation.append(hidden_layer[i + 1][1])
            # layers.append(self.dropout)

        last_layer = nn.Linear(hidden_layer[-1][0], output_features)
        init.xavier_normal_(last_layer.weight)
        self.output_layer = last_layer
        # layers.append(last_layer)

        if last_activation == "lambda":
            layers.append(AdaptiveSigmoid())
            self.last_activation = AdaptiveSigmoid()
        elif last_activation == None:
            self.last_activation = None
        else:
            self.last_activation = last_activation

        self.layers = nn.ModuleList(layers)
        self.activation = nn.ModuleList(activation)

    def forward(self, x):
        for idx, (layer, activation) in enumerate(zip(self.layers, self.activation)):
            x = self.dropout(activation(layer(x)))
        x = self.output_layer(x)

        if self.last_activation is not None:
            x = self.last_activation(x)
        return x


# @define(slots=True)
# class GM_NN_Model:
#     parameters: dict = field(factory=dict)
#     criterion: list = field(factory=list, init=True)
#     n_components: int = field(default=4, init=True)
#     bias: bool = field(default=False, init=True)
#     init_params: str = field(default="random", init=True)
#     base_dir: str = field(init=True, default=check_base_dir(BASE_DATA_DIR))
#     seed: int = field(default=42, init=True)

#     nn_model: NeuralNet = NeuralNet(NeuralNetworkModular, nn.MSELoss)
#     nn_best_params: dict = field(factory=dict)
#     gmm_target_x: np.ndarray = field(init=True, default=np.array(None))
#     gmm_target_y: np.ndarray = field(init=True, default=np.array(None))
#     history: dict = field(factory=dict)

#     def __init__(
#         self,
#         parameters: dict = {},
#         n_components: int = 4,
#         bias: bool = False,
#         init_params: str = "random",
#         base_dir: list = None,
#         seed: int = None,
#     ):
#         if base_dir is None:
#             base_dir = check_base_dir(BASE_DATA_DIR)
#         else:
#             base_dir = check_base_dir(base_dir)

#         if isinstance(parameters, list) == False and parameters.get("criterion") == None:
#             raise ValueError("Please specify a valid criterion!")
#         criterion = parameters.get("criterion")

#         if isinstance(criterion, list) == False:
#             criterion = [criterion]

#         self.__attrs_init__(
#             parameters,
#             criterion=criterion,
#             n_components=n_components,
#             bias=bias,
#             init_params=init_params,
#             base_dir=base_dir,
#             seed=seed,
#         )

#     def __attrs_post_init__(self):
#         if self.seed is not None:

#             set_seed(self.seed)
#             # torch.manual_seed(self.seed)
#         # setup the Gaussian Mixture Model:

#     def fit(
#         self,
#         X,
#         Y: np.ndarray,
#         search_type: str = None,
#         n_jobs: int = 1,
#         device: str = "cpu",
#         patience: int = 20,
#         early_stop: bool = False,
#     ):
#         if device == "auto" or device == None:
#             device = "cuda" if torch.cuda.is_available() else "cpu"

#         callbacks = []
#         # callbacks.append(
#         #     EpochScoring(scoring="r2", lower_is_better=False),
#         # )
#         # callbacks.append(LRScheduler(policy=torch.optim.lr_scheduler.LinearLR))

#         # wandb_run = wandb.init()

#         if early_stop != False and early_stop in ["valid_loss", "r2"]:

#             callbacks.append(
#                 EarlyStopping(
#                     monitor=early_stop,
#                     patience=patience,
#                     load_best=False,
#                     lower_is_better=False,
#                 ),
#             )

#         if search_type == "auto":
#             for value in self.parameters.values():
#                 if isinstance(value, list):
#                     search_type = "gridsearch"
#                     break

#         if search_type == None:
#             new_dict = {}
#             for key, value in self.parameters.items():
#                 if isinstance(value, list) and len(value) > 0:
#                     new_dict[key] = value[0]
#                 else:
#                     new_dict[key] = value

#             # wandb_run.config.update(new_dict)
#             # callbacks.append(WandbLogger(wandb_run))

#             self.nn_model = NeuralNet(
#                 NeuralNetworkModular,
#                 **new_dict,
#                 verbose=1,
#                 device=device,
#                 module__device=device,
#                 module__input_features=1,
#                 callbacks=callbacks,
#             )

#         elif search_type == "gridsearch":
#             new_dict = {}
#             for key, value in self.parameters.items():
#                 if not isinstance(value, list):
#                     new_dict[key] = [value]
#                 else:
#                     new_dict[key] = value

#             nn_model_net = NeuralNet(
#                 NeuralNetworkModular,
#                 self.criterion[0],
#                 verbose=0,
#                 device=device,
#                 module__device=device,
#                 module__input_features=X.shape[1],
#                 callbacks=callbacks,
#             )

#             self.nn_model = GridSearchCV(
#                 nn_model_net,
#                 new_dict,
#                 refit="r2",
#                 cv=5,
#                 verbose=3,
#                 n_jobs=n_jobs,
#                 scoring=["r2"],
#             )

#         self.nn_model.fit(
#             torch.tensor(X, dtype=torch.float32),
#             torch.tensor(Y, dtype=torch.float32),
#         )
#         # print("HISTORY", self.nn_model.history)

#         # print(self.nn_model.history)

#         if search_type == "gridsearch":
#             self.nn_best_params = self.nn_model.best_params_
#             print(self.nn_model.cv_results_)

#         else:
#             self.nn_best_params = new_dict
#             self.history = self.nn_model.history

#         return self.nn_model, self.gmm_target_y

#     def predict(self, X):
#         return self.nn_model.predict(torch.tensor(X, dtype=torch.float32))


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
