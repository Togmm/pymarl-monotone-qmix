import math
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


class _CenteredSoftplus(nn.Module):
    def __init__(self, beta=1.0):
        super(_CenteredSoftplus, self).__init__()
        self.beta = beta
        self.softplus = nn.Softplus(beta=beta)

    def forward(self, x):
        return self.softplus(x) - math.log(2.0) / self.beta


def _activation_slope(activation, x):
    """Return the pointwise derivative used for layer diagnostics."""
    if isinstance(activation, _CenteredSoftplus):
        return th.sigmoid(activation.beta * x)
    if isinstance(activation, nn.Softplus):
        return th.sigmoid(activation.beta * x)
    if isinstance(activation, nn.ReLU):
        return x.gt(0).to(x.dtype)
    if isinstance(activation, nn.ELU):
        return th.where(x.gt(0), th.ones_like(x), activation.alpha * th.exp(x))
    if isinstance(activation, nn.CELU):
        return th.where(
            x.gt(0), th.ones_like(x), th.exp(x / activation.alpha)
        )
    if isinstance(activation, nn.SELU):
        selu_alpha = 1.6732632423543772
        selu_scale = 1.0507009873554805
        return th.where(
            x.gt(0),
            th.full_like(x, selu_scale),
            selu_scale * selu_alpha * th.exp(x),
        )
    if isinstance(activation, nn.Tanh):
        return 1.0 - th.tanh(x).pow(2)
    if isinstance(activation, _Identity):
        return th.ones_like(x)
    return None


def _activation(name, softplus_beta=1.0):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "celu":
        return nn.CELU()
    if name == "selu":
        return nn.SELU()
    if name == "softplus":
        return nn.Softplus(beta=softplus_beta)
    if name == "centered_softplus":
        return _CenteredSoftplus(beta=softplus_beta)
    if name == "silu":
        return _SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError("Unknown activation '{}'".format(name))


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


class AMCOMonotonicLinear(nn.Linear):
    """Monotonic linear layer from AMCO-UniPD/monotonic."""

    def __init__(self, in_features, out_features, bias=True, pre_activation=None):
        super(AMCOMonotonicLinear, self).__init__(
            in_features, out_features, bias=bias
        )
        self.act = pre_activation if pre_activation is not None else _Identity()
        self.diagnostics_enabled = False

    def forward(self, x):
        if self.diagnostics_enabled:
            x_detached = x.detach()
            self._last_preactivation_mean = x_detached.mean(dim=-1)
            self._last_preactivation_std = x_detached.std(
                dim=-1, unbiased=False
            )
            self._last_nonpositive_fraction = x_detached.le(0).float().mean(
                dim=-1
            )
            slopes = _activation_slope(self.act, x_detached)
            if slopes is not None:
                self._last_activation_slope_mean = slopes.mean(dim=-1)
                sorted_slopes = slopes.sort(dim=-1).values
                p10_index = int(round(0.1 * (sorted_slopes.size(-1) - 1)))
                self._last_activation_slope_p10 = sorted_slopes[
                    ..., p10_index
                ]
        w_pos = self.weight.clamp(min=0.0)
        w_neg = self.weight.clamp(max=0.0)
        x_pos = F.linear(self.act(x), w_pos, self.bias)
        x_neg = F.linear(self.act(-x), w_neg, None)
        return x_pos + x_neg


class AMCOPartialMonotonicInputLayer(nn.Module):
    """Input layer that is monotone in Q and free in state features."""

    def __init__(
        self,
        q_features,
        state_features,
        out_features,
        state_input_scale=1.0,
        bias=True,
    ):
        super(AMCOPartialMonotonicInputLayer, self).__init__()
        self.q_features = q_features
        self.state_features = state_features
        self.out_features = out_features
        self.state_input_scale = state_input_scale

        self.q_weight = nn.Parameter(th.Tensor(out_features, q_features))
        self.state_weight = nn.Parameter(th.Tensor(out_features, state_features))
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.q_weight, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.state_weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in = self.q_features + self.state_features
            bound = 1.0 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, agent_qs, state_features, return_components=False):
        q_w_pos = self.q_weight.clamp(min=0.0)
        q_w_neg = self.q_weight.clamp(max=0.0)
        q_term = (
            F.linear(agent_qs, q_w_pos, None)
            + F.linear(-agent_qs, q_w_neg, None)
        )
        state_term = self.state_input_scale * F.linear(
            state_features, self.state_weight, self.bias
        )
        output = q_term + state_term
        if return_components:
            return output, q_term, state_term
        return output


class AMCOMonotoneMixer(nn.Module):
    """AMCO-style partially monotonic mixer.

    The state is encoded by an unconstrained MLP z = g(s). The first mixer
    layer is monotone only in [Q_1, ..., Q_n] and free in z; all later layers
    preserve monotonicity through AMCO monotonic linear maps.
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
        self.mono_softplus_beta = float(
            getattr(args, "amco_mono_softplus_beta", 1.0)
        )
        self.state_value_dim = getattr(args, "amco_state_value_dim", 32)
        self.state_value_activation = getattr(
            args, "amco_state_value_activation", "relu"
        )
        self.state_input_scale = _map_override(
            args, "amco_state_input_scale", 1.0
        )
        self.q_residual_scale = _map_override(
            args, "amco_q_residual_scale", 0.0
        )
        self.q_residual_final_scale = _map_override(
            args, "amco_q_residual_final_scale", self.q_residual_scale
        )
        self.q_residual_anneal_steps = _map_override(
            args, "amco_q_residual_anneal_steps", 0
        )
        self.q_residual_mode = _map_override(
            args, "amco_q_residual_mode", "sum"
        )
        self.train_step = 0
        self.diagnostics_enabled = False
        self._last_diagnostic_tensors = None

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
            "softplus",
            "centered_softplus",
            "tanh",
        ):
            raise ValueError(
                "amco_mono_activation must be globally monotone to preserve IGM"
            )
        if (
            self.mono_activation_name.lower()
            in ("softplus", "centered_softplus")
            and (
                not np.isfinite(self.mono_softplus_beta)
                or self.mono_softplus_beta <= 0
            )
        ):
            raise ValueError("amco_mono_softplus_beta must be finite and positive")
        if self.q_residual_scale < 0:
            raise ValueError(
                "amco_q_residual_scale must be non-negative to preserve IGM"
            )
        if self.q_residual_final_scale < 0:
            raise ValueError(
                "amco_q_residual_final_scale must be non-negative to preserve IGM"
            )
        if self.q_residual_anneal_steps < 0:
            raise ValueError("amco_q_residual_anneal_steps must be non-negative")
        if self.q_residual_mode not in ("sum", "mean"):
            raise ValueError("amco_q_residual_mode must be 'sum' or 'mean'")

        self.state_encoder = self._build_state_encoder()
        self.input_layer = AMCOPartialMonotonicInputLayer(
            self.n_agents,
            self.state_embed_dim,
            self.mono_hidden_dim,
            state_input_scale=self.state_input_scale,
        )
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
        for _ in range(self.mono_depth - 2):
            layers.append(
                AMCOMonotonicLinear(
                    self.mono_hidden_dim,
                    self.mono_hidden_dim,
                    pre_activation=_activation(
                        self.mono_activation_name,
                        softplus_beta=self.mono_softplus_beta,
                    ),
                )
            )

        layers.append(
            AMCOMonotonicLinear(
                self.mono_hidden_dim,
                1,
                pre_activation=_activation(
                    self.mono_activation_name,
                    softplus_beta=self.mono_softplus_beta,
                ),
            )
        )
        return nn.Sequential(*layers)

    def set_train_step(self, t_env):
        self.train_step = int(t_env)

    def _current_q_residual_scale(self):
        if self.q_residual_anneal_steps <= 0:
            return self.q_residual_scale
        progress = min(
            float(self.train_step) / float(self.q_residual_anneal_steps),
            1.0,
        )
        return (
            self.q_residual_scale
            + progress
            * (self.q_residual_final_scale - self.q_residual_scale)
        )

    def _q_residual(self, agent_qs):
        if self.q_residual_mode == "mean":
            residual = agent_qs.mean(dim=1, keepdim=True)
        else:
            residual = agent_qs.sum(dim=1, keepdim=True)
        return self._current_q_residual_scale() * residual

    def set_diagnostics_enabled(self, enabled):
        self.diagnostics_enabled = bool(enabled)
        self._last_diagnostic_tensors = None
        for layer in self.monotone_net:
            if hasattr(layer, "diagnostics_enabled"):
                layer.diagnostics_enabled = self.diagnostics_enabled

    @staticmethod
    def _masked_mean(values, mask):
        return (values * mask).sum() / mask.sum().clamp(min=1.0)

    def get_diagnostics(self, mask=None):
        """Return detached AMCO branch statistics from the latest forward."""
        tensors = self._last_diagnostic_tensors
        if tensors is None:
            return {}

        reference = tensors["q_term_power"]
        if mask is None:
            mask = reference.new_ones(reference.shape)
        else:
            mask = mask.detach().reshape_as(reference).to(reference.dtype)

        eps = 1e-8
        q_rms = self._masked_mean(tensors["q_term_power"], mask).sqrt()
        state_rms = self._masked_mean(
            tensors["state_term_power"], mask
        ).sqrt()
        mixing_rms = self._masked_mean(
            tensors["mixing_output_power"], mask
        ).sqrt()
        value_rms = self._masked_mean(
            tensors["state_value_power"], mask
        ).sqrt()

        diagnostics = {
            "amco_q_term_rms": q_rms,
            "amco_state_term_rms": state_rms,
            "amco_state_to_q_ratio": state_rms / (q_rms + eps),
            "amco_mixing_output_rms": mixing_rms,
            "amco_state_value_rms": value_rms,
            "amco_v_to_m_ratio": value_rms / (mixing_rms + eps),
        }
        if "pre_activation_nonpositive_fraction" in tensors:
            diagnostics["amco_pre_activation_nonpositive_frac"] = self._masked_mean(
                tensors["pre_activation_nonpositive_fraction"], mask
            )
        for key, value in tensors.items():
            if key.startswith("layer_"):
                diagnostics["amco_{}".format(key)] = self._masked_mean(
                    value, mask
                )
        return diagnostics

    @staticmethod
    def _gradient_stats(parameter):
        if parameter.grad is None:
            zero = parameter.detach().new_zeros(())
            return zero, zero
        grad = parameter.grad.detach()
        norm = grad.norm()
        rms = norm / float(parameter.numel()) ** 0.5
        return norm, rms

    def get_gradient_diagnostics(self):
        """Return raw post-backward mixer gradient diagnostics."""
        q_norm, q_rms = self._gradient_stats(self.input_layer.q_weight)
        state_norm, state_rms = self._gradient_stats(
            self.input_layer.state_weight
        )
        encoder_parameters = list(self.state_encoder.parameters())
        encoder_sq = q_norm.new_zeros(())
        encoder_count = 0
        for parameter in encoder_parameters:
            if parameter.grad is not None:
                encoder_sq = encoder_sq + parameter.grad.detach().pow(2).sum()
                encoder_count += parameter.numel()
        encoder_norm = encoder_sq.sqrt()
        encoder_rms = encoder_norm / max(1, encoder_count) ** 0.5
        return {
            "amco_input_q_weight_grad_norm": q_norm,
            "amco_input_state_weight_grad_norm": state_norm,
            "amco_input_q_weight_grad_rms": q_rms,
            "amco_input_state_weight_grad_rms": state_rms,
            "amco_input_state_to_q_grad_rms_ratio": state_rms
            / (q_rms + 1e-8),
            "amco_state_encoder_grad_norm": encoder_norm,
            "amco_state_encoder_grad_rms": encoder_rms,
        }

    def get_state_credit_diagnostics(
        self, agent_qs, states, credit_grads, mask=None
    ):
        """Measure credit changes after a deterministic state permutation.

        States are rolled across batch elements at each time index while the
        individual Q values stay fixed. Only transitions valid in both the
        original and rolled episodes contribute to the metric.
        """
        if agent_qs.size(0) < 2:
            zero = agent_qs.detach().new_zeros(())
            return {
                "amco_state_credit_delta": zero,
                "amco_state_credit_share_delta": zero,
            }

        if mask is None:
            mask = agent_qs.new_ones(agent_qs.shape[:-1] + (1,))
        pair_mask = mask.detach() * mask.detach().roll(1, dims=0)
        q_counterfactual = agent_qs.detach().requires_grad_(True)
        saved_diagnostics = self._last_diagnostic_tensors
        diagnostics_enabled = self.diagnostics_enabled
        self.set_diagnostics_enabled(False)
        try:
            counterfactual_output = self(
                q_counterfactual, states.detach().roll(1, dims=0)
            )
            counterfactual_grads = th.autograd.grad(
                outputs=counterfactual_output,
                inputs=q_counterfactual,
                grad_outputs=pair_mask,
                create_graph=False,
            )[0].detach()
        finally:
            self.set_diagnostics_enabled(diagnostics_enabled)
            self._last_diagnostic_tensors = saved_diagnostics

        valid = pair_mask.detach().expand_as(credit_grads).gt(0)
        actual = credit_grads.detach()
        counterfactual = counterfactual_grads
        if not valid.any():
            zero = actual.new_zeros(())
            return {
                "amco_state_credit_delta": zero,
                "amco_state_credit_share_delta": zero,
            }

        delta = (actual - counterfactual).abs().sum(dim=-1)
        denominator = actual.abs().sum(dim=-1).clamp(min=1e-8)
        state_delta = (delta / denominator)[pair_mask.squeeze(-1).gt(0)]

        actual_positive = actual.clamp(min=0.0)
        counterfactual_positive = counterfactual.clamp(min=0.0)
        actual_share = actual_positive / actual_positive.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-8)
        counterfactual_share = counterfactual_positive / counterfactual_positive.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-8)
        share_delta = (
            actual_share - counterfactual_share
        ).abs().sum(dim=-1)[pair_mask.squeeze(-1).gt(0)]
        return {
            "amco_state_credit_delta": state_delta.mean(),
            "amco_state_credit_share_delta": share_delta.mean(),
        }

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, self.n_agents)

        state_features = self.state_encoder(states)
        hidden, q_term, state_term = self.input_layer(
            agent_qs, state_features, return_components=True
        )
        q_residual = self._q_residual(agent_qs)
        mixing_output = self.monotone_net(hidden)
        state_value = self.state_value(states)
        q_tot = mixing_output + state_value + q_residual

        if self.diagnostics_enabled:
            steps = q_tot.size(0) // bs
            diagnostic_tensors = {
                "q_term_power": (
                    q_term.detach().pow(2).mean(dim=-1).view(bs, steps)
                ),
                "state_term_power": (
                    state_term.detach().pow(2).mean(dim=-1).view(bs, steps)
                ),
                "mixing_output_power": (
                    mixing_output.detach().pow(2).squeeze(-1).view(bs, steps)
                ),
                "state_value_power": (
                    state_value.detach().pow(2).squeeze(-1).view(bs, steps)
                ),
            }
            nonpositive = [
                layer._last_nonpositive_fraction
                for layer in self.monotone_net
                if hasattr(layer, "_last_nonpositive_fraction")
            ]
            if nonpositive:
                diagnostic_tensors["pre_activation_nonpositive_fraction"] = (
                    th.stack(nonpositive, dim=1)
                    .mean(dim=1)
                    .view(bs, steps)
                )
            for layer_index, layer in enumerate(self.monotone_net):
                layer_stats = (
                    ("preactivation_mean", "_last_preactivation_mean"),
                    ("preactivation_std", "_last_preactivation_std"),
                    ("nonpositive_frac", "_last_nonpositive_fraction"),
                    ("activation_slope_mean", "_last_activation_slope_mean"),
                    ("activation_slope_p10", "_last_activation_slope_p10"),
                )
                for suffix, attribute in layer_stats:
                    if hasattr(layer, attribute):
                        diagnostic_tensors[
                            "layer_{}_{}".format(layer_index, suffix)
                        ] = getattr(layer, attribute).view(bs, steps)
            self._last_diagnostic_tensors = diagnostic_tensors
        return q_tot.view(bs, -1, 1)
