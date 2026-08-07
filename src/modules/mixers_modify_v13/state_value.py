import torch.nn as nn


def _activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError("Unknown state-value activation '{}'".format(name))


class StateValueNetwork(nn.Module):
    """Unconstrained V(s) residual used alongside a monotone interaction term."""

    def __init__(self, state_dim, hidden_dim=32, activation="relu"):
        super(StateValueNetwork, self).__init__()
        if hidden_dim < 1:
            raise ValueError("state-value hidden dimension must be positive")

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _activation(activation),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states):
        return self.net(states)
