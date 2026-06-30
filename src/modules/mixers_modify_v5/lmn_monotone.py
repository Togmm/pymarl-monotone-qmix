import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.mixers.state_value import StateValueNetwork


class _SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


def _activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "silu":
        return _SiLU()
    if name == "selu":
        return nn.SELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError("Unknown activation '{}'".format(name))


class LMNGroupSort(nn.Module):
    """GroupSort activation used by expressive Lipschitz networks."""

    def __init__(self, group_size=2):
        super(LMNGroupSort, self).__init__()
        if group_size < 2:
            raise ValueError("lmn_group_size must be at least 2")
        self.group_size = group_size

    def forward(self, x):
        if x.size(1) % self.group_size != 0:
            raise ValueError(
                "LMN hidden width must be divisible by lmn_group_size"
            )
        grouped = x.view(
            x.size(0), x.size(1) // self.group_size, self.group_size
        )
        return grouped.sort(dim=2)[0].view_as(x)


class LMNNormedLinear(nn.Module):
    """Linear layer with differentiable forward-time norm constraints.

    ``one-inf`` bounds each matrix element and is used in the first layer.
    ``inf`` bounds the absolute row sum and is used in subsequent layers.
    This matches the construction used by niklasnolte/monotonic_tests.
    """

    def __init__(self, in_features, out_features, max_norm, kind, bias=True):
        super(LMNNormedLinear, self).__init__()

        if kind not in ("one-inf", "inf"):
            raise ValueError("Unknown LMN norm kind '{}'".format(kind))
        if max_norm <= 0:
            raise ValueError("LMN layer max_norm must be positive")

        self.in_features = in_features
        self.out_features = out_features
        self.max_norm = max_norm
        self.kind = kind

        self.weight = nn.Parameter(th.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(float(self.in_features))
        self.weight.data.uniform_(-bound, bound)
        if self.bias is not None:
            self.bias.data.uniform_(-bound, bound)

    def normed_weight(self):
        if self.kind == "one-inf":
            norms = self.weight.abs()
        else:
            norms = self.weight.abs().sum(dim=1, keepdim=True)

        divisor = th.clamp(norms / self.max_norm, min=1.0)
        return self.weight / divisor

    def forward(self, x):
        return F.linear(x, self.normed_weight(), self.bias)


class LMNLipschitzNetwork(nn.Module):
    """GroupSort network with a certified L1 Lipschitz constant."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        hidden_depth,
        group_size,
        lipschitz_const,
    ):
        super(LMNLipschitzNetwork, self).__init__()

        if hidden_depth < 1:
            raise ValueError("lmn_hidden_depth must be at least 1")
        if hidden_dim < 1:
            raise ValueError("lmn_hidden_dim must be positive")
        if hidden_dim % group_size != 0:
            raise ValueError(
                "lmn_hidden_dim must be divisible by lmn_group_size"
            )
        if lipschitz_const <= 0:
            raise ValueError("lmn_lipschitz_const must be positive")

        self.lipschitz_const = lipschitz_const
        linear_depth = hidden_depth + 1
        per_layer_lip = lipschitz_const ** (1.0 / linear_depth)

        layers = [
            LMNNormedLinear(
                input_dim,
                hidden_dim,
                max_norm=per_layer_lip,
                kind="one-inf",
            ),
            LMNGroupSort(group_size),
        ]
        for _ in range(hidden_depth - 1):
            layers.extend(
                [
                    LMNNormedLinear(
                        hidden_dim,
                        hidden_dim,
                        max_norm=per_layer_lip,
                        kind="inf",
                    ),
                    LMNGroupSort(group_size),
                ]
            )
        layers.append(
            LMNNormedLinear(
                hidden_dim,
                1,
                max_norm=per_layer_lip,
                kind="inf",
            )
        )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LMNMonotoneMixer(nn.Module):
    """Expressive Lipschitz Monotonic Network mixer.

    Let g(Q, z(s)) be L1-Lipschitz with constant lambda. The mixer is

        Q_tot = g(Q, z(s)) + lambda * sum_i Q_i + V(s).

    Hence dQ_tot/dQ_i >= 0 for every agent while the state remains an
    unconstrained input to g and to the state-value baseline.
    """

    def __init__(self, args):
        super(LMNMonotoneMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.state_embed_dim = getattr(args, "lmn_state_embed_dim", 64)
        self.state_encoder_depth = getattr(
            args, "lmn_state_encoder_depth", 1
        )
        self.state_activation = getattr(args, "lmn_state_activation", "silu")

        self.hidden_dim = getattr(args, "lmn_hidden_dim", 64)
        self.hidden_depth = getattr(args, "lmn_hidden_depth", 2)
        self.group_size = getattr(args, "lmn_group_size", 2)
        self.lipschitz_const = getattr(
            args, "lmn_lipschitz_const", 0.1
        )
        self.residual_scale = getattr(args, "lmn_residual_scale", 1.0)

        self.state_value_dim = getattr(args, "lmn_state_value_dim", 32)
        self.state_value_activation = getattr(
            args, "lmn_state_value_activation", "relu"
        )

        if self.state_embed_dim < 1:
            raise ValueError("lmn_state_embed_dim must be positive")
        if self.state_encoder_depth < 1:
            raise ValueError("lmn_state_encoder_depth must be at least 1")
        if self.residual_scale < 1.0:
            raise ValueError(
                "lmn_residual_scale must be at least 1 to certify monotonicity"
            )

        self.state_encoder = self._build_state_encoder()
        self.lipschitz_network = LMNLipschitzNetwork(
            input_dim=self.n_agents + self.state_embed_dim,
            hidden_dim=self.hidden_dim,
            hidden_depth=self.hidden_depth,
            group_size=self.group_size,
            lipschitz_const=self.lipschitz_const,
        )
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
            layers.append(_activation(self.state_activation))
            in_dim = self.state_embed_dim
        return nn.Sequential(*layers)

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, self.n_agents)

        state_features = self.state_encoder(states)
        network_inputs = th.cat((agent_qs, state_features), dim=1)
        lipschitz_output = self.lipschitz_network(network_inputs)
        monotone_residual = (
            self.residual_scale
            * self.lipschitz_const
            * agent_qs.sum(dim=1, keepdim=True)
        )
        # q_tot = (
        #     lipschitz_output
        #     + monotone_residual
        #     + self.state_value(states)
        # )
        q_tot = lipschitz_output + monotone_residual
        return q_tot.view(bs, -1, 1)
