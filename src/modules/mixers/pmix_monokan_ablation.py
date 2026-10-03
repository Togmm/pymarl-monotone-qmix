"""State-path ablations for the MonoKAN/PMIX mixer.

The full ``monokan`` mixer is ``V(s) + M(q, s)``.  This module provides the
matched B1/B2 controls ``M(q)`` and ``V(s) + M(q)``, plus B3 ``M(q, s)``
without ``V(s)``, using the same calibrated Q path and MonoKAN core.
"""

import numpy as np
import torch as th
import torch.nn as nn

from modules.mixers.monokan_monotone import (
    MonoKANCore,
    MonoKANMonotoneMixer,
    _map_override,
)
from modules.mixers.state_value import StateValueNetwork


class _MonoKANAblationMixer(nn.Module):
    def __init__(self, args, use_state_value):
        super(_MonoKANAblationMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.use_state_value = bool(use_state_value)

        self.hidden_dim = _map_override(args, "monokan_hidden_dim", 32)
        self.num_intervals = _map_override(args, "monokan_grid", 7)
        self.grid_min = getattr(args, "monokan_grid_min", -1.0)
        self.grid_max = getattr(args, "monokan_grid_max", 1.0)
        self.noise_scale = getattr(args, "monokan_noise_scale", 0.02)
        self.min_increment = getattr(args, "monokan_min_increment", 1e-4)
        self.q_temperature = _map_override(args, "monokan_q_temperature", 1.0)
        self.q_residual_scale = _map_override(args, "monokan_q_residual_scale", 0.0)
        self.q_residual_mode = _map_override(args, "monokan_q_residual_mode", "sum")
        self.q_residual_input = _map_override(args, "monokan_q_residual_input", "raw")
        self.state_value_dim = getattr(args, "monokan_state_value_dim", 32)
        self.state_value_activation = getattr(args, "monokan_state_value_activation", "relu")

        if self.hidden_dim < 1 or self.num_intervals < 1:
            raise ValueError("MonoKAN hidden dimension and grid must be positive")
        if self.q_temperature <= 0:
            raise ValueError("monokan_q_temperature must be positive")
        if self.q_residual_scale < 0:
            raise ValueError("monokan_q_residual_scale must be non-negative")
        if self.q_residual_mode not in ("sum", "mean"):
            raise ValueError("monokan_q_residual_mode must be 'sum' or 'mean'")
        if self.q_residual_input not in ("raw", "tanh"):
            raise ValueError("monokan_q_residual_input must be 'raw' or 'tanh'")

        # state_feature_dim=0 is intentional: the first KAN layer is
        # monotone in every Q and has no state input or unused state weights.
        self.monokan = MonoKANCore(
            n_agents=self.n_agents,
            state_feature_dim=0,
            hidden_dim=self.hidden_dim,
            num_intervals=self.num_intervals,
            grid_min=self.grid_min,
            grid_max=self.grid_max,
            noise_scale=self.noise_scale,
            min_increment=self.min_increment,
        )
        if self.use_state_value:
            self.state_value = StateValueNetwork(
                self.state_dim,
                hidden_dim=self.state_value_dim,
                activation=self.state_value_activation,
            )

    def _q_residual(self, agent_qs, q_features):
        residual_input = q_features if self.q_residual_input == "tanh" else agent_qs
        residual = (
            residual_input.mean(dim=1, keepdim=True)
            if self.q_residual_mode == "mean"
            else residual_input.sum(dim=1, keepdim=True)
        )
        return self.q_residual_scale * residual

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, self.n_agents)
        q_features = th.tanh(agent_qs / self.q_temperature)
        empty_state = q_features.new_empty((q_features.size(0), 0))
        q_tot = self.monokan(q_features, empty_state)
        q_tot = q_tot + self._q_residual(agent_qs, q_features)
        if self.use_state_value:
            q_tot = q_tot + self.state_value(states)
        return q_tot.view(bs, -1, 1)


class MonoKANQOnlyMixer(_MonoKANAblationMixer):
    """B1: a state-free monotone MonoKAN mixer ``M(q)``."""

    def __init__(self, args):
        super(MonoKANQOnlyMixer, self).__init__(args, use_state_value=False)


class MonoKANQValueMixer(_MonoKANAblationMixer):
    """B2: an additive state baseline ``V(s) + M(q)``."""

    def __init__(self, args):
        super(MonoKANQValueMixer, self).__init__(args, use_state_value=True)


class MonoKANQStateMixer(MonoKANMonotoneMixer):
    """B3: state changes the MonoKAN Q interaction, without ``V(s)``."""

    def __init__(self, args):
        # Reuse the full PMIX-KAN state encoder and interaction path, while
        # removing only its additive state-value branch.
        super(MonoKANQStateMixer, self).__init__(args)
        del self.state_value

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, self.n_agents)
        q_features = th.tanh(agent_qs / self.q_temperature)
        state_features = th.tanh(self.state_encoder(states))
        q_tot = self.monokan(q_features, state_features)
        q_tot = q_tot + self._q_residual(agent_qs, q_features)
        return q_tot.view(bs, -1, 1)
