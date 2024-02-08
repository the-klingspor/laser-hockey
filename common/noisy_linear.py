import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Mu and Sigma for weights
        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_sigma = nn.Parameter(torch.empty((out_features, in_features)))

        # Mu and Sigma for bias
        self.bias_mu = nn.Parameter(torch.empty((out_features,)))
        self.bias_sigma = nn.Parameter(torch.empty((out_features,)))

        # Noise
        self.register_buffer("weight_epsilon", torch.empty((out_features, in_features)))
        self.register_buffer("bias_epsilon", torch.empty((out_features,)))

        mu_range = 1 / math.sqrt(self.in_features)
        weight_sigma_value = 0.5 / math.sqrt(self.in_features)
        bias_sigma_value = 0.5 / math.sqrt(self.out_features)

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(weight_sigma_value)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(bias_sigma_value)

        self.reset_noise()

    def reset_noise(self) -> None:
        epsilon_in = self.get_noise(self.in_features)
        epsilon_out = self.get_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def get_noise(self, size: int) -> Tensor:
        noise = torch.randn(size, device=self.weight_mu.device)
        return noise.sign() * noise.abs().sqrt()

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)
