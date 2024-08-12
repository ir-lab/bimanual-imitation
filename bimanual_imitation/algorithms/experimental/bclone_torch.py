import torch
import torch.nn as nn


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, n_layers, activation="relu"):
        super().__init__()

        self.log_stds = nn.Parameter(torch.zeros(action_dim))

        # Add Input layer
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden_dim))

        for layer_idx in range(n_layers):
            if activation == "relu":
                activation_obj = nn.ReLU()
            elif activation == "tanh":
                activation_obj = nn.Tanh()
            self.net.add_module(f"{activation}_{layer_idx}", activation_obj)

            if layer_idx == n_layers - 1:
                # Final Hidden layer
                self.net.add_module(f"linear_{layer_idx}", nn.Linear(hidden_dim, action_dim))
            else:
                # Middle Hidden layer
                self.net.add_module(f"linear_{layer_idx}", nn.Linear(hidden_dim, hidden_dim))

    def forward(self, sample):
        action_means = self.net(sample)
        action_stds = torch.ones_like(action_means) * self.log_stds.exp()
        return action_means, action_stds
