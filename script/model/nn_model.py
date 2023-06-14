import torch.nn as nn
import torch
import torch.nn.functional as F


class NeuralNetworkModular(nn.Module):
    def __init__(
        self,
        input_features,
        output_features=2,
        dropout=0.5,
        num_units=20,
        activation=nn.ReLU,
        n_layer=1,
        type_layer="decrease",
        softmax=True,
    ):
        super().__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dropout = nn.Dropout(dropout)
        self.layers = []
        self.layers.append(nn.Linear(input_features, num_units, dtype=torch.float64, device=device))
        for i in range(n_layer - 1):
            if type_layer == "decrease":
                self.layers.append(
                    nn.Linear(
                        int(num_units / (2**i)), int(num_units / (2 ** (i + 1))), dtype=torch.float64, device=device
                    )
                )

            else:
                self.layers.append(
                    nn.Linear(
                        int(num_units * (2**i), int(num_units * (2 ** (i + 1))), dtype=torch.float64, device=device)
                    )
                )

        if type_layer == "decrease":
            self.output_layer = nn.Linear(int(num_units / (2 ** (n_layer - 1))), output_features, dtype=torch.float64)
        else:
            self.output_layer = nn.Linear(int(num_units * (2 ** (n_layer - 1))), output_features, dtype=torch.float64)

        self.activation = activation
        self.softmax_v = softmax

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)

        x = self.output_layer(x)
        if self.softmax_v:
            x = F.softmax(x, dim=-1)

        return x
