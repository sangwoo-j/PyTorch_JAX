# ------------------------------- #
# Make a MLP
# u_hat = MLP(input_dim=1, hidden_dim=2, num_layers=2, output_dim=1, activation).to(device)
# u_hat(t, x1, x2, ...)

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=2, num_layers=2, output_dim=1, activation = "tanh"):
        super().__init__()

        self.num_layers = num_layers

        layers = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        activations = {"tanh": nn.Tanh(),
                       "relu": nn.ReLU(),
                       "sigmoid": nn.Sigmoid(),
                       "leaky_relu": nn.LeakyReLU()}
        if activation not in activations:
            raise ValueError(f"Unsupported activation function: {activation}")
        self.activation = activations[activation]

    def forward(self, *inputs):
        inputs = torch.cat(inputs, dim=1)

        x = self.activation(self.hidden_layers[0](inputs))
        for i in range(1, self.num_layers):
            x = self.activation(self.hidden_layers[i](x))
        output = self.output_layer(x)

        return output