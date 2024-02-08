import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Module

from common.noisy_linear import NoisyLinear


class MLP(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        output_dim: int,
        activation_function: str,
        use_noisy_linear: bool,
        squash: bool = False,
    ) -> None:
        super().__init__()

        if activation_function == "relu":
            activation = nn.ReLU
        elif activation_function == "leaky_relu":
            activation = nn.LeakyReLU
        elif activation_function == "swish":
            activation = nn.SiLU
        elif activation_function == "mish":
            activation = nn.Mish
        else:
            raise NotImplementedError

        self.layers = nn.ModuleList()
        last_dim = input_dim

        self.squash = squash

        # Create hidden layers
        for hidden_dim in hidden_layers:
            self.layers.append(nn.Linear(last_dim, hidden_dim))
            self.layers.append(activation())
            last_dim = hidden_dim

        linear = NoisyLinear if use_noisy_linear else nn.Linear

        # Output layer
        self.layers.append(linear(last_dim, output_dim))

    def reset_noise(self) -> None:
        for layer in self.layers:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)

        if self.squash:
            x = F.tanh(x)

        return x
