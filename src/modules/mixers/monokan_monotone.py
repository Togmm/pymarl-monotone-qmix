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


def _inverse_softplus(value):
    return math.log(math.exp(value) - 1.0)


def _map_name(args):
    env_args = getattr(args, "env_args", {}) or {}
    if isinstance(env_args, dict):
        return env_args.get("map_name")
    return getattr(env_args, "map_name", None)


def _map_override(args, key, default):
    value = getattr(args, key, default)
    by_map = getattr(args, "{}_by_map".format(key), None)
    map_name = _map_name(args)
    if isinstance(by_map, dict) and map_name in by_map:
        return by_map[map_name]
    return value


class MonoKANHermiteLayer(nn.Module):
    """KAN layer using certified monotonic cubic Hermite edge splines.

    Monotone edge values are represented by cumulative positive increments.
    Their derivatives are bounded by the adjacent secant slopes according to
    the Fritsch-Carlson sufficient condition. Free edges retain unconstrained
    knot values, derivatives, and scales.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        num_intervals,
        monotone_inputs,
        grid_min=-1.0,
        grid_max=1.0,
        noise_scale=0.02,
        min_increment=1e-4,
    ):
        super(MonoKANHermiteLayer, self).__init__()

        if num_intervals < 1:
            raise ValueError("monokan_grid must be at least 1")
        if grid_min >= grid_max:
            raise ValueError("MonoKAN grid range must be strictly ordered")
        if len(monotone_inputs) != in_dim:
            raise ValueError("monotone_inputs must contain one flag per input")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_intervals = num_intervals
        self.num_knots = num_intervals + 1
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.grid_step = (grid_max - grid_min) / float(num_intervals)
        self.min_increment = min_increment

        mask = np.asarray(monotone_inputs).astype(np.float32)
        self.register_buffer(
            "monotone_mask", th.FloatTensor(mask).view(1, in_dim, 1)
        )
        grid = th.linspace(grid_min, grid_max, steps=self.num_knots)
        self.register_buffer("grid", grid)

        increment_value = max(self.grid_step * 0.5, self.min_increment)
        self.knot_start = nn.Parameter(
            th.Tensor(out_dim, in_dim, 1).fill_(
                -0.5 * num_intervals * increment_value
            )
        )
        increment_init = _inverse_softplus(increment_value)
        self.raw_increments = nn.Parameter(
            th.Tensor(out_dim, in_dim, num_intervals)
        )
        self.free_knot_values = nn.Parameter(
            th.Tensor(out_dim, in_dim, self.num_knots)
        )
        self.raw_slopes = nn.Parameter(
            th.Tensor(out_dim, in_dim, self.num_knots)
        )
        self.raw_scale_base = nn.Parameter(th.Tensor(out_dim, in_dim))
        self.raw_scale_spline = nn.Parameter(th.Tensor(out_dim, in_dim))
        self.bias = nn.Parameter(th.zeros(out_dim))

        self.raw_increments.data.normal_(increment_init, noise_scale)
        self.free_knot_values.data.normal_(0.0, noise_scale)
        self.raw_slopes.data.normal_(0.0, noise_scale)
        edge_mask = th.FloatTensor(mask).view(1, in_dim).expand(
            out_dim, in_dim
        )
        scale_target = 0.5 / math.sqrt(float(in_dim))
        monotone_scale_init = _inverse_softplus(scale_target)
        scale_init = (
            edge_mask * monotone_scale_init
            + (1.0 - edge_mask) * scale_target
        )
        self.raw_scale_base.data.copy_(
            scale_init + th.randn(out_dim, in_dim) * noise_scale
        )
        self.raw_scale_spline.data.copy_(
            scale_init + th.randn(out_dim, in_dim) * noise_scale
        )

    def _minimum(self, left, right):
        return 0.5 * (left + right - th.abs(left - right))

    def _edge_parameters(self):
        increments = F.softplus(self.raw_increments) + self.min_increment
        monotone_values = th.cat(
            (
                self.knot_start,
                self.knot_start + th.cumsum(increments, dim=2),
            ),
            dim=2,
        )
        knot_values = (
            self.monotone_mask * monotone_values
            + (1.0 - self.monotone_mask) * self.free_knot_values
        )

        secants = (
            monotone_values[:, :, 1:] - monotone_values[:, :, :-1]
        ) / self.grid_step
        slope_factor = 3.0 / math.sqrt(2.0)
        endpoint_left = slope_factor * secants[:, :, :1]
        endpoint_right = slope_factor * secants[:, :, -1:]
        if self.num_intervals > 1:
            interior = slope_factor * self._minimum(
                secants[:, :, :-1], secants[:, :, 1:]
            )
            slope_limit = th.cat(
                (endpoint_left, interior, endpoint_right), dim=2
            )
        else:
            slope_limit = th.cat((endpoint_left, endpoint_right), dim=2)

        monotone_slopes = th.sigmoid(self.raw_slopes) * slope_limit
        slopes = (
            self.monotone_mask * monotone_slopes
            + (1.0 - self.monotone_mask) * self.raw_slopes
        )

        edge_mask = self.monotone_mask.squeeze(2)
        scale_base = (
            edge_mask * F.softplus(self.raw_scale_base)
            + (1.0 - edge_mask) * self.raw_scale_base
        )
        scale_spline = (
            edge_mask * F.softplus(self.raw_scale_spline)
            + (1.0 - edge_mask) * self.raw_scale_spline
        )
        return knot_values, slopes, scale_base, scale_spline

    def _evaluate_splines(self, x, knot_values, slopes):
        batch_size = x.size(0)
        scaled = (x - self.grid_min) / self.grid_step
        interval = th.floor(scaled).long()
        interval = th.clamp(
            interval, min=0, max=self.num_intervals - 1
        )
        t = scaled - interval.float()

        interval = interval.unsqueeze(1).expand(
            batch_size, self.out_dim, self.in_dim
        )
        interval_right = interval + 1
        values = knot_values.unsqueeze(0).expand(
            batch_size, self.out_dim, self.in_dim, self.num_knots
        )
        derivatives = slopes.unsqueeze(0).expand_as(values)
        gather_left = interval.unsqueeze(3)
        gather_right = interval_right.unsqueeze(3)

        y0 = th.gather(values, 3, gather_left).squeeze(3)
        y1 = th.gather(values, 3, gather_right).squeeze(3)
        m0 = th.gather(derivatives, 3, gather_left).squeeze(3)
        m1 = th.gather(derivatives, 3, gather_right).squeeze(3)

        t = t.unsqueeze(1).expand(
            batch_size, self.out_dim, self.in_dim
        )
        t2 = t * t
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        return (
            y0 * h00
            + m0 * self.grid_step * h10
            + y1 * h01
            + m1 * self.grid_step * h11
        )

    def forward(self, x):
        knot_values, slopes, scale_base, scale_spline = (
            self._edge_parameters()
        )
        spline = self._evaluate_splines(x, knot_values, slopes)
        base = (th.sigmoid(x) - 0.5).unsqueeze(1).expand(
            x.size(0), self.out_dim, self.in_dim
        )
        edge_outputs = (
            scale_base.unsqueeze(0) * base
            + scale_spline.unsqueeze(0) * spline
        )
        return edge_outputs.sum(dim=2) + self.bias.unsqueeze(0)


class MonoKANCore(nn.Module):
    """Two-layer partially monotonic KAN used as a value mixer."""

    def __init__(
        self,
        n_agents,
        state_feature_dim,
        hidden_dim,
        num_intervals,
        grid_min,
        grid_max,
        noise_scale,
        min_increment,
    ):
        super(MonoKANCore, self).__init__()

        input_dim = n_agents + state_feature_dim
        first_mask = [1] * n_agents + [0] * state_feature_dim
        self.input_layer = MonoKANHermiteLayer(
            input_dim,
            hidden_dim,
            num_intervals,
            first_mask,
            grid_min=grid_min,
            grid_max=grid_max,
            noise_scale=noise_scale,
            min_increment=min_increment,
        )
        self.output_layer = MonoKANHermiteLayer(
            hidden_dim,
            1,
            num_intervals,
            [1] * hidden_dim,
            grid_min=grid_min,
            grid_max=grid_max,
            noise_scale=noise_scale,
            min_increment=min_increment,
        )

    def forward(self, agent_qs, state_features):
        x = th.cat((agent_qs, state_features), dim=1)
        hidden = th.tanh(self.input_layer(x))
        return self.output_layer(hidden)


class MonoKANMonotoneMixer(nn.Module):
    """Certified partial-monotone MonoKAN mixer replacing QMIX."""

    def __init__(self, args):
        super(MonoKANMonotoneMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.state_embed_dim = _map_override(args, "monokan_state_embed_dim", 32)
        self.state_encoder_depth = getattr(
            args, "monokan_state_encoder_depth", 1
        )
        self.state_activation = getattr(
            args, "monokan_state_activation", "silu"
        )
        self.hidden_dim = _map_override(args, "monokan_hidden_dim", 32)
        self.num_intervals = _map_override(args, "monokan_grid", 7)
        self.grid_min = getattr(args, "monokan_grid_min", -1.0)
        self.grid_max = getattr(args, "monokan_grid_max", 1.0)
        self.noise_scale = getattr(args, "monokan_noise_scale", 0.02)
        self.min_increment = getattr(
            args, "monokan_min_increment", 1e-4
        )
        self.q_temperature = _map_override(args, "monokan_q_temperature", 1.0)
        self.q_residual_scale = _map_override(
            args, "monokan_q_residual_scale", 0.0
        )
        self.q_residual_mode = _map_override(
            args, "monokan_q_residual_mode", "sum"
        )
        self.state_value_dim = getattr(args, "monokan_state_value_dim", 32)
        self.state_value_activation = getattr(
            args, "monokan_state_value_activation", "relu"
        )

        if self.state_embed_dim < 1:
            raise ValueError("monokan_state_embed_dim must be positive")
        if self.state_encoder_depth < 1:
            raise ValueError("monokan_state_encoder_depth must be at least 1")
        if self.hidden_dim < 1:
            raise ValueError("monokan_hidden_dim must be positive")
        if self.q_temperature <= 0:
            raise ValueError("monokan_q_temperature must be positive")
        if self.q_residual_scale < 0:
            raise ValueError(
                "monokan_q_residual_scale must be non-negative to preserve IGM"
            )
        if self.q_residual_mode not in ("sum", "mean"):
            raise ValueError("monokan_q_residual_mode must be 'sum' or 'mean'")

        self.state_encoder = self._build_state_encoder()
        self.monokan = MonoKANCore(
            n_agents=self.n_agents,
            state_feature_dim=self.state_embed_dim,
            hidden_dim=self.hidden_dim,
            num_intervals=self.num_intervals,
            grid_min=self.grid_min,
            grid_max=self.grid_max,
            noise_scale=self.noise_scale,
            min_increment=self.min_increment,
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

    def _q_residual(self, agent_qs):
        if self.q_residual_mode == "mean":
            residual = agent_qs.mean(dim=1, keepdim=True)
        else:
            residual = agent_qs.sum(dim=1, keepdim=True)
        return self.q_residual_scale * residual

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, self.n_agents)

        q_features = th.tanh(agent_qs / self.q_temperature)
        state_features = th.tanh(self.state_encoder(states))
        q_tot = self.monokan(q_features, state_features)
        q_residual = self._q_residual(agent_qs)
        q_tot = q_tot + self.state_value(states) + q_residual
        return q_tot.view(bs, -1, 1)
