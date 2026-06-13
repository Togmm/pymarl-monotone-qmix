import numpy as np
import torch as th
import torch.nn as nn


class _Identity(nn.Module):
    def forward(self, x):
        return x


class _SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


def _truncated_normal_(tensor, mean=0.0, std=1.0):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


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


class SMMSmoothMonotonicNN(nn.Module):
    """Smooth Min-Max Monotonic Network from christian-igel/SMM.

    This keeps the original SMM parameterisation: each group applies a smooth
    max over affine functions, and the output applies a smooth min over groups.
    The mask selects which input variables must have non-negative weights.
    """

    def __init__(
        self,
        n,
        K,
        h_K,
        mask=None,
        b_z=1.0,
        b_t=1.0,
        beta=-1.0,
        transform="exp",
        scale_beta=False,
    ):
        super(SMMSmoothMonotonicNN, self).__init__()

        self.n = n
        self.K = K
        self.h_K = h_K
        self.beta_init = beta
        if scale_beta:
            self.b_z = b_z * np.exp(self.beta_init)
            self.b_t = b_t * np.exp(self.beta_init)
        else:
            self.b_z = b_z
            self.b_t = b_t

        self.gamma = nn.Parameter(th.zeros(1), requires_grad=True)
        self.beta = nn.Parameter(th.ones(1), requires_grad=True)
        self.z = nn.ParameterList(
            [nn.Parameter(th.ones(h_K, n), requires_grad=True) for _ in range(K)]
        )
        self.t = nn.ParameterList(
            [nn.Parameter(th.ones(h_K), requires_grad=True) for _ in range(K)]
        )

        if mask is None:
            self.mask = None
            self.mask_inv = None
        else:
            mask = np.asarray(mask).astype(np.float32)
            assert mask.shape == (n,)
            self.register_buffer("mask", th.FloatTensor(mask))
            self.register_buffer("mask_inv", 1.0 - self.mask)

        self.transform = transform
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.K):
            _truncated_normal_(self.z[i], std=self.b_z)
            _truncated_normal_(self.t[i], std=self.b_t)
        nn.init.constant_(self.beta, self.beta_init)

    def soft_max(self, a):
        return th.logsumexp(a, dim=1)

    def soft_min(self, a):
        return -th.logsumexp(-a, dim=1)

    def _positive_weight(self, z):
        if self.transform == "exp":
            return th.exp(z)
        if self.transform == "abs":
            return th.abs(z)
        if self.transform == "explin":
            return th.where(z > 1.0, z, th.exp(z - 1.0))
        if self.transform == "sqr":
            return z * z
        raise ValueError("Unknown SMM transform '{}'".format(self.transform))

    def forward(self, x):
        if x.dim() == 1:
            x = x.reshape(-1, 1)

        group_outputs = []
        for i in range(self.K):
            w = self._positive_weight(self.z[i])
            if self.mask is not None:
                w = self.mask * w + self.mask_inv * self.z[i]

            a = th.matmul(x, w.t()) + self.t[i]
            g = self.soft_max(a).unsqueeze(1)
            group_outputs.append(g)

        y = th.cat(group_outputs, dim=1)
        y = self.soft_min(y) / th.exp(self.beta) + self.gamma
        return y


class SMMMonotoneMixer(nn.Module):
    """SMM partially monotonic mixer for QMIX-style value factorisation.

    Inputs to SMM are [Q_1, ..., Q_n, z(s)]. The SMM mask constrains only the
    agent-Q dimensions, so Q_tot is monotone in each Q_i and unconstrained in
    the state embedding z(s).
    """

    def __init__(self, args):
        super(SMMMonotoneMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.state_embed_dim = getattr(args, "smm_state_embed_dim", 64)
        self.state_encoder_depth = getattr(args, "smm_state_encoder_depth", 1)
        self.state_activation_name = getattr(args, "smm_state_activation", "silu")

        self.K = getattr(args, "smm_K", 6)
        self.h_K = getattr(args, "smm_h_K", 6)
        self.b_z = getattr(args, "smm_b_z", 1.0)
        self.b_t = getattr(args, "smm_b_t", 1.0)
        self.beta = getattr(args, "smm_beta", -1.0)
        self.transform = getattr(args, "smm_transform", "exp")
        self.scale_beta = getattr(args, "smm_scale_beta", False)

        if self.state_encoder_depth < 0:
            raise ValueError("smm_state_encoder_depth must be non-negative")
        if self.state_embed_dim < 0:
            raise ValueError("smm_state_embed_dim must be non-negative")
        if self.state_embed_dim > 0 and self.state_encoder_depth < 1:
            raise ValueError(
                "smm_state_encoder_depth must be at least 1 when "
                "smm_state_embed_dim is positive"
            )

        self.state_encoder = self._build_state_encoder()
        smm_input_dim = self.n_agents + self._state_feature_dim()
        monotone_mask = np.zeros(smm_input_dim)
        monotone_mask[: self.n_agents] = 1

        self.smm = SMMSmoothMonotonicNN(
            n=smm_input_dim,
            K=self.K,
            h_K=self.h_K,
            mask=monotone_mask,
            b_z=self.b_z,
            b_t=self.b_t,
            beta=self.beta,
            transform=self.transform,
            scale_beta=self.scale_beta,
        )

    def _state_feature_dim(self):
        return self.state_embed_dim if self.state_embed_dim > 0 else self.state_dim

    def _build_state_encoder(self):
        if self.state_embed_dim == 0:
            return _Identity()

        layers = []
        in_dim = self.state_dim
        for _ in range(self.state_encoder_depth):
            layers.append(nn.Linear(in_dim, self.state_embed_dim))
            layers.append(_activation(self.state_activation_name))
            in_dim = self.state_embed_dim
        return nn.Sequential(*layers)

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, self.n_agents)

        state_features = self.state_encoder(states)
        mixer_inputs = th.cat([agent_qs, state_features], dim=-1)
        q_tot = self.smm(mixer_inputs)
        return q_tot.view(bs, -1, 1)
