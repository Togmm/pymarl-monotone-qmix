import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.mixers.state_value import StateValueNetwork


class _Identity(nn.Module):
    def forward(self, x):
        return x


class _SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


def _activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "celu":
        return nn.CELU()
    if name == "selu":
        return nn.SELU()
    if name == "silu":
        return _SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError("Unknown activation '{}'".format(name))


class AMCOMonotonicLinear(nn.Linear):
    """Monotonic linear layer from AMCO-UniPD/monotonic."""

    def __init__(self, in_features, out_features, bias=True, pre_activation=None):
        super(AMCOMonotonicLinear, self).__init__(
            in_features, out_features, bias=bias
        )
        self.act = pre_activation if pre_activation is not None else _Identity()

    def forward(self, x):
        w_pos = self.weight.clamp(min=0.0)
        w_neg = self.weight.clamp(max=0.0)
        x_pos = F.linear(self.act(x), w_pos, self.bias)
        x_neg = F.linear(self.act(-x), w_neg, self.bias)
        return x_pos + x_neg


class AMCOMonotoneMixer(nn.Module):
    """AMCO-style partially monotonic mixer.

    The state is encoded by an unconstrained MLP z = g(s), while the final
    AMCO monotonic MLP maps [Q_1, ..., Q_n, z] to Q_tot. Since z is a free
    function of state, the mixer is constrained to be monotone only in the
    agent Q inputs required for IGM.
    """

    def __init__(self, args):
        super(AMCOMonotoneMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.state_embed_dim = getattr(args, "amco_state_embed_dim", 128)
        self.mono_hidden_dim = getattr(args, "amco_mono_hidden_dim", 128)
        self.mono_depth = getattr(args, "amco_mono_depth", 4)
        self.state_encoder_depth = getattr(args, "amco_state_encoder_depth", 2)
        self.state_activation_name = getattr(args, "amco_state_activation", "silu")
        self.mono_activation_name = getattr(args, "amco_mono_activation", "selu")
        self.state_value_dim = getattr(args, "amco_state_value_dim", 32)
        self.state_value_activation = getattr(
            args, "amco_state_value_activation", "relu"
        )

        if self.mono_depth < 4:
            raise ValueError(
                "amco_mono_depth must be at least 4 to follow AMCO's "
                "universal approximation recommendation"
            )
        if self.state_encoder_depth < 1:
            raise ValueError("amco_state_encoder_depth must be at least 1")
        if self.mono_activation_name.lower() not in (
            "relu",
            "elu",
            "celu",
            "selu",
            "tanh",
        ):
            raise ValueError(
                "amco_mono_activation must be globally monotone to preserve IGM"
            )

        self.state_encoder = self._build_state_encoder()
        self.monotone_net = self._build_monotone_net()
        self.state_value = StateValueNetwork(
            self.state_dim,
            hidden_dim=self.state_value_dim,
            activation=self.state_value_activation,
        )

    def _build_state_encoder(self):
        layers = []
        in_dim = self.state_dim
        for _ in range(self.state_encoder_depth):
            layers.append(nn.Linear(in_dim, self.state_embed_dim))
            layers.append(_activation(self.state_activation_name))
            in_dim = self.state_embed_dim
        return nn.Sequential(*layers)

    def _build_monotone_net(self):
        layers = []
        in_dim = self.n_agents + self.state_embed_dim

        layers.append(
            AMCOMonotonicLinear(
                in_dim,
                self.mono_hidden_dim,
                pre_activation=_Identity(),
            )
        )

        for _ in range(self.mono_depth - 2):
            layers.append(
                AMCOMonotonicLinear(
                    self.mono_hidden_dim,
                    self.mono_hidden_dim,
                    pre_activation=_activation(self.mono_activation_name),
                )
            )

        layers.append(
            AMCOMonotonicLinear(
                self.mono_hidden_dim,
                1,
                pre_activation=_activation(self.mono_activation_name),
            )
        )
        return nn.Sequential(*layers)

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, self.n_agents)

        state_features = self.state_encoder(states)
        mixer_inputs = th.cat([agent_qs, state_features], dim=-1)
        q_tot = self.monotone_net(mixer_inputs) + self.state_value(states)
        # q_tot = self.monotone_net(mixer_inputs)
        return q_tot.view(bs, -1, 1)
