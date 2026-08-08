"""
Noisy Linear Layer with Factorized Gaussian Noise (Fortunato et al., 2017 Noisy Networks for Exploration).

Architectural Design & Behavior Notes:
-------------------------------------
- Parameterization:
  Replaces standard linear weights W and bias b with learned mean (mu) and noise scaling (sigma) parameters:
      W = weight_mu + weight_sigma * weight_epsilon
      b = bias_mu + bias_sigma * bias_epsilon

- Factorized Noise Generation:
  Uses factorized Gaussian noise to reduce random number generation from O(p * q) to O(p + q):
      f(x) = sgn(x) * sqrt(|x|)
      weight_epsilon = f(epsilon_out) (outer) f(epsilon_in)
      bias_epsilon = f(epsilon_out)

- Training vs Evaluation Mode (`self.training`):
  During training (`model.train()`), noise buffers are applied (`weight_mu + weight_sigma * weight_epsilon`).
  During evaluation (`model.eval()`), noise is removed and purely mean weights (`weight_mu`, `bias_mu`) are used.

- Resampling Noise:
  `reset_noise()` should be invoked per optimization update step (or environment step depending on protocol)
  to sample fresh independent noise variables.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """
    Factorized Noisy Linear layer for exploration in deep RL (Fortunato et al., 2017).

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        sigma_init (float): Initial value for noise scaling parameter sigma. Defaults to 0.5.
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        # Non-learnable Noise Buffers (epsilon)
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initializes learnable mu and sigma parameters according to Fortunato et al. (2017)."""
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Applies real-valued sign square-root factorized noise function f(x) = sgn(x) * sqrt(|x|)."""
        x = torch.randn(size, device=self.weight_mu.device, dtype=self.weight_mu.dtype)
        return x.sign().mul_(x.abs().sqrt_())

    # TODO: what is considered a step in noisy nets dqn? update step? each minibatch? env step?
    def reset_noise(self):
        """Samples fresh noise tensors epsilon_in and epsilon_out and updates noise buffers."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input feature tensor of shape [..., in_features].

        Returns:
            torch.Tensor: Linear projection tensor of shape [..., out_features].
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)
