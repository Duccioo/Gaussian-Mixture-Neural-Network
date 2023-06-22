import torch.nn as nn
import torch
from skorch import NeuralNet
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np


# ---


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
        input_features=1,
        output_features=1,
        dropout=0.5,
        num_units=10,
        activation=nn.ReLU(),
        last_activation=False,
        n_layer=2,
        type_layer="decrease",
        device="cpu",
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers = []
        self.layers.append(nn.Linear(input_features, num_units, device=device))
        for i in range(n_layer - 1):
            if type_layer == "decrease":
                self.layers.append(
                    nn.Linear(int(num_units / (2**i)), int(num_units / (2 ** (i + 1))), device=device)
                )

            else:
                self.layers.append(
                    nn.Linear(int(num_units * (2**i)), int(num_units * (2 ** (i + 1))), device=device)
                )

        if type_layer == "decrease":
            self.output_layer = nn.Linear(int(num_units / (2 ** (n_layer - 1))), output_features, device=device)
        else:
            self.output_layer = nn.Linear(int(num_units * (2 ** (n_layer - 1))), output_features, device=device)

        self.activation = activation

        if last_activation == "lambda":
            self.last_activation = AdaptiveSigmoid()
        else:
            self.last_activation = last_activation

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)

        x = self.output_layer(x)

        if self.last_activation:
            x = self.last_activation(x)

        return x


def NerualNetwork_model(
    parameters: dict, search: str = None, device: str = "auto", n_jobs: int = 4, seed=42, **kwargs
):
    """
    Function for create the model of the Network with gridsearch and randomsearch possibilities.

    Args:
        parameters (dict):
        search (str, optional): Option:[gridsearch, randomsearch]. Defaults to None.
        device (str, optional): Option:[auto, cpu, cuda]. Defaults to "auto".
        n_jobs (int, optional): Description: number of cores to use in the gridsearch and randomsearch. Defaults to -1.
        seed (int, optional): Description: Seed for the initialization of the parameters

    Returns:
        a Neural Network model in the type of sklearn model
    """
    torch.manual_seed(seed)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if type(parameters["criterion"]) == list or search == "gridsearch":
        model = NeuralNet(NeuralNetworkModular, parameters["criterion"][0], **kwargs, verbose=0, device=device, module__device=device,)
        model = GridSearchCV(
            model,
            parameters,
            refit="r2",
            cv=5,
            verbose=3,
            n_jobs=n_jobs,
            scoring=["r2", "max_error", "explained_variance", "neg_mean_absolute_percentage_error"],
            
        )

    elif search == "randomsearch":
        model = NeuralNet(NeuralNetworkModular, device=device, **kwargs)
        model = RandomizedSearchCV(model, parameters, refit=True, cv=5, verbose=50, n_jobs=n_jobs)

    else:
        model = NeuralNet(NeuralNetworkModular, criterion=parameters["criterion"], **kwargs, verbose=0, device=device)

    return model


if __name__ == "__main__":
    num_samples = 1000
    test_size = 0.2
    random_state = 42

    # Generate random number for a random dataset
    np.random.seed(random_state)
    x = np.random.uniform(0, 1, num_samples)
    y = x**2

    # split the dataset into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    x_train = x_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # create the model
    params = {"criterion": nn.L1Loss}
    model = NerualNetwork_model(
        params,
        lr=0.01,
        max_epochs=100,
        batch_size=128,
        module__n_layer=2,
        module__activation=nn.Tanh(),
        module__num_units=500,
    )

    # train the model
    model.fit(torch.Tensor(x_train), torch.Tensor(y_train))

    # test the model
    out = model.predict(torch.Tensor(x_test))
    print(out[0:5].reshape(-1).round(3))
    print(y_test[0:5].reshape(-1).round(3))
