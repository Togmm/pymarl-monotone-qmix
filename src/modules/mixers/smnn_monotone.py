import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def truncated_normal_(tensor, mean=0.0, std=1.0):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def _as_tuple(value, default):
    if value is None:
        return tuple(default)
    if isinstance(value, int):
        return (value,)
    return tuple(value)


def _activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "silu":
        return nn.SiLU()
    if name == "selu":
        return nn.SELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError("Unknown activation '{}'".format(name))


class SMNNActivationLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SMNNActivationLayer, self).__init__()
        self.weight = nn.Parameter(th.empty((in_features, out_features)))
        self.bias = nn.Parameter(th.empty(out_features))

    def forward(self, x):
        raise NotImplementedError


class SMNNExpUnit(SMNNActivationLayer):
    def __init__(self, in_features, out_features):
        super(SMNNExpUnit, self).__init__(in_features, out_features)
        nn.init.uniform_(self.weight, a=-20.0, b=2.0)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        out = x @ th.exp(self.weight) + self.bias
        return 0.99 * th.clip(out, 0, 1) + 0.01 * out


class SMNNReLUUnit(SMNNActivationLayer):
    def __init__(self, in_features, out_features):
        super(SMNNReLUUnit, self).__init__(in_features, out_features)
        nn.init.xavier_uniform_(self.weight)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        return F.relu(x @ self.weight + self.bias)


class SMNNConfluenceUnit(SMNNActivationLayer):
    def __init__(self, in_features, out_features):
        super(SMNNConfluenceUnit, self).__init__(in_features, out_features)
        nn.init.xavier_uniform_(self.weight)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        out = x @ self.weight + self.bias
        return 0.99 * th.clip(out, 0, 1) + 0.01 * out


class SMNNFCLayer(SMNNActivationLayer):
    def __init__(self, in_features, out_features):
        super(SMNNFCLayer, self).__init__(in_features, out_features)
        truncated_normal_(self.weight, mean=-10.0, std=3)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        return x @ th.exp(self.weight) + self.bias


class SMNNCore(nn.Module):
    """Scalable Monotonic Neural Network adapted from retna319/SMNN."""

    def __init__(self, mono_size, non_mono_size, exp_sizes, relu_sizes, conf_sizes):
        super(SMNNCore, self).__init__()

        if not (len(exp_sizes) == len(relu_sizes) == len(conf_sizes)):
            raise ValueError(
                "SMNN exp, relu, and confluence branches must have equal depth"
            )
        if len(exp_sizes) == 0:
            raise ValueError("SMNN branch depth must be at least 1")

        self.mono_size = mono_size
        self.non_mono_size = non_mono_size
        self.exp_sizes = exp_sizes
        self.relu_sizes = relu_sizes
        self.conf_sizes = conf_sizes

        self.exp_units = nn.ModuleList(
            [
                SMNNExpUnit(
                    mono_size if i == 0 else exp_sizes[i - 1] + conf_sizes[i - 1],
                    exp_sizes[i],
                )
                for i in range(len(exp_sizes))
            ]
        )
        self.relu_units = nn.ModuleList(
            [
                SMNNReLUUnit(
                    non_mono_size if i == 0 else relu_sizes[i - 1],
                    relu_sizes[i],
                )
                for i in range(len(relu_sizes))
            ]
        )
        self.conf_units = nn.ModuleList(
            [
                SMNNConfluenceUnit(
                    non_mono_size if i == 0 else relu_sizes[i - 1],
                    conf_sizes[i],
                )
                for i in range(len(conf_sizes))
            ]
        )

        final_in = exp_sizes[-1] + conf_sizes[-1] + relu_sizes[-1]
        self.fc_layer = SMNNFCLayer(final_in, 1)

    def forward(self, x_mono, x_non_mono):
        for i in range(len(self.exp_sizes)):
            if i == 0:
                exp_output = self.exp_units[i](x_mono)
                conf_output = self.conf_units[i](x_non_mono)
                relu_output = self.relu_units[i](x_non_mono)
            else:
                exp_output = self.exp_units[i](exp_output)
                conf_output = self.conf_units[i](relu_output)
                relu_output = self.relu_units[i](relu_output)
            exp_output = th.cat([exp_output, conf_output], dim=1)

        return self.fc_layer(th.cat([exp_output, relu_output], dim=1))


class SMNNMonotoneMixer(nn.Module):
    """SMNN-style partially monotonic mixer for value factorisation.

    Agent Q values are the monotone features. The global state is treated as
    non-monotone context, optionally through a free state encoder. This keeps
    Q_tot monotone in each Q_i for IGM while letting state affect the mixing
    relation without a monotonicity constraint.
    """

    def __init__(self, args):
        super(SMNNMonotoneMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.state_embed_dim = getattr(args, "smnn_state_embed_dim", 0)
        self.state_encoder_depth = getattr(args, "smnn_state_encoder_depth", 1)
        self.state_activation = getattr(args, "smnn_state_activation", "silu")

        self.exp_sizes = _as_tuple(
            getattr(args, "smnn_exp_unit_size", None), (128, 128)
        )
        self.relu_sizes = _as_tuple(
            getattr(args, "smnn_relu_unit_size", None), (32, 32)
        )
        self.conf_sizes = _as_tuple(
            getattr(args, "smnn_conf_unit_size", None), (64, 64)
        )

        if self.state_embed_dim and self.state_embed_dim > 0:
            self.state_encoder = self._build_state_encoder()
            non_mono_size = self.state_embed_dim
        else:
            self.state_encoder = nn.Identity()
            non_mono_size = self.state_dim

        self.smnn = SMNNCore(
            mono_size=self.n_agents,
            non_mono_size=non_mono_size,
            exp_sizes=self.exp_sizes,
            relu_sizes=self.relu_sizes,
            conf_sizes=self.conf_sizes,
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
        q_tot = self.smnn(agent_qs, state_features)
        return q_tot.view(bs, -1, 1)
