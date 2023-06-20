import torch.nn as nn
import torch
from skorch import NeuralNet
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import numpy as np
from sklearn.model_selection import train_test_split


class NeuralNetworkModular(nn.Module):
    def __init__(
        self,
        input_features=1,
        output_features=1,
        dropout=0.5,
        num_units=10,
        activation=nn.ReLU(),
        last_activation=False,
        n_layer=1,
        type_layer="decrease",
    ):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
                    nn.Linear(int(num_units * (2**i), int(num_units * (2 ** (i + 1))), device=device))
                )

        if type_layer == "decrease":
            self.output_layer = nn.Linear(int(num_units / (2 ** (n_layer - 1))), output_features)
        else:
            self.output_layer = nn.Linear(int(num_units * (2 ** (n_layer - 1))), output_features)

        self.activation = activation
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


def NerualNetwork_model(parameters: dict, search: str = None, device: str = "auto", n_jobs: int = -1, **kwargs):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if type(parameters["criterion"]) == list or search == "gridsearch":
        model = NeuralNet(NeuralNetworkModular)
        model = GridSearchCV(model, parameters, refit=True, cv=5, verbose=50, n_jobs=n_jobs, **kwargs)

    elif search == "randomsearch":
        model = NeuralNet(NeuralNetworkModular)
        model = RandomizedSearchCV(model, parameters, refit=True, cv=5, verbose=50, n_jobs=n_jobs, **kwargs)

    else:
        model = NeuralNet(NeuralNetworkModular, criterion=parameters["criterion"], **kwargs)
        print(model.criterion)

    return model


if __name__ == "__main__":
    params = {"criterion": nn.L1Loss}

    num_samples = 1000
    test_size = 0.2
    random_state = 42

    # Genera dati casuali secondo la distribuzione di probabilità specificata
    np.random.seed(random_state)
    x = np.random.uniform(0, 1, num_samples)
    y = x**2

    # Dividi i dati in training set e test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    x_train.reshape(-1, 1)
    x_test.reshape(-1, 1)
    y_train.reshape(-1, 1)
    y_test.reshape(-1, 1)

    model = NerualNetwork_model(
        params,
        lr=0.01,
        max_epochs=100,
        batch_size=128,
        module__n_layer=2,
        module__activation=nn.Tanh(),
        module__num_units=500,
    )

    model.fit(torch.Tensor(x_train), torch.Tensor(y_train))

    out = model.predict(torch.Tensor(x_test))
    print(out[0:5].reshape(-1).round(3))
    print(y_test[0:5].reshape(-1).round(3))
