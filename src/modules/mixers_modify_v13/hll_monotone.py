import itertools
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
    """Return x such that softplus(x) is value for a positive scalar."""
    if value <= 0:
        raise ValueError("softplus target must be positive")
    return math.log(math.expm1(value))


class LearnedMonotoneQCalibrator(nn.Module):
    """Shared strictly monotone affine-sigmoid calibration for HLL inputs."""

    def __init__(
        self,
        init_shift=0.0,
        init_scale=1.0,
        min_scale=0.25,
    ):
        super(LearnedMonotoneQCalibrator, self).__init__()
        if init_scale <= min_scale:
            raise ValueError(
                "hll_q_calibrator_init_scale must exceed "
                "hll_q_calibrator_min_scale"
            )
        if min_scale <= 0:
            raise ValueError("hll_q_calibrator_min_scale must be positive")

        self.shift = nn.Parameter(th.tensor(float(init_shift)))
        self.raw_scale = nn.Parameter(
            th.tensor(_inverse_softplus(init_scale - min_scale))
        )
        self.min_scale = float(min_scale)

    def scale(self):
        return F.softplus(self.raw_scale) + self.min_scale

    def forward(self, q_values):
        scale = self.scale()
        coordinates = th.sigmoid((q_values - self.shift) / scale)
        return coordinates, scale


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def _lattice_sizes(value, n_agents):
    if isinstance(value, int):
        sizes = (value,) * n_agents
    else:
        sizes = tuple(value)
    if len(sizes) != n_agents:
        raise ValueError(
            "hll_lattice_size must be an int or contain one value per agent"
        )
    if any(size < 2 for size in sizes):
        raise ValueError("Every HLL lattice size must be at least 2")
    return sizes


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


class HLLAuxiliaryNetwork(nn.Module):
    """State network producing the HLL vertex interpolation parameters."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        depth,
        activation,
        output_bias_init,
    ):
        super(HLLAuxiliaryNetwork, self).__init__()

        if depth < 1:
            raise ValueError("hll_aux_depth must be at least 1")

        layers = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_activation(activation))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.output_bias_init = output_bias_init
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
        self.network[-1].bias.data.fill_(self.output_bias_init)

    def forward(self, x):
        return th.sigmoid(self.network(x))


class HLLHierarchicalLattice(nn.Module):
    """Vectorised Hierarchical Lattice Layer adapted from IBM/pmlayer."""

    def __init__(self, lattice_sizes, auxiliary_network, max_vertices):
        super(HLLHierarchicalLattice, self).__init__()

        self.lattice_sizes = tuple(int(size) for size in lattice_sizes)
        self.n_monotone = len(self.lattice_sizes)
        self.num_vertices = _product(self.lattice_sizes)
        self.binary_lattice = all(size == 2 for size in self.lattice_sizes)
        self.diagnostics_enabled = False
        self._last_vertex_values = None
        self._last_probabilities = None
        if self.num_vertices > max_vertices:
            raise ValueError(
                "HLL lattice has {} vertices, exceeding hll_max_vertices={}. "
                "Reduce hll_lattice_size.".format(
                    self.num_vertices, max_vertices
                )
            )

        self.auxiliary_network = auxiliary_network

        coefficients = []
        for i in range(self.n_monotone):
            coefficients.append(_product(self.lattice_sizes[i + 1 :]))
        self.register_buffer(
            "mesh_size", th.LongTensor(self.lattice_sizes)
        )
        self.register_buffer(
            "index_coefficients", th.LongTensor(coefficients)
        )

        corners = list(itertools.product((0, 1), repeat=self.n_monotone))
        self.register_buffer("corners", th.LongTensor(corners))

        coordinates = list(
            itertools.product(
                *[range(size) for size in self.lattice_sizes]
            )
        )
        edge_lower_indices = []
        edge_upper_indices = []
        edge_dimensions = []
        edge_level_buckets = []
        max_lower_level = max(sum(self.lattice_sizes) - self.n_monotone - 1, 1)
        for coordinate in coordinates:
            lower_index = self._coordinate_to_index(coordinate)
            lower_level = sum(coordinate)
            level_bucket = min(
                int(3.0 * lower_level / float(max_lower_level + 1)), 2
            )
            for dim, size in enumerate(self.lattice_sizes):
                if coordinate[dim] >= size - 1:
                    continue
                upper = list(coordinate)
                upper[dim] += 1
                edge_lower_indices.append(lower_index)
                edge_upper_indices.append(
                    self._coordinate_to_index(tuple(upper))
                )
                edge_dimensions.append(dim)
                edge_level_buckets.append(level_bucket)
        # Diagnostic-only topology is derived from lattice_sizes and excluded
        # from state_dict so existing checkpoints remain loadable.
        self.edge_lower_indices = th.LongTensor(edge_lower_indices)
        self.edge_upper_indices = th.LongTensor(edge_upper_indices)
        self.edge_dimensions = th.LongTensor(edge_dimensions)
        self.edge_level_buckets = th.LongTensor(edge_level_buckets)

        levels = {}
        for coordinate in coordinates:
            levels.setdefault(sum(coordinate), []).append(coordinate)

        self.level_buffer_names = []
        for level, level_coordinates in sorted(levels.items()):
            indices = []
            lower_indices = []
            lower_masks = []
            for coordinate in level_coordinates:
                indices.append(self._coordinate_to_index(coordinate))
                predecessors = []
                predecessor_mask = []
                for dim in range(self.n_monotone):
                    if coordinate[dim] > 0:
                        lower = list(coordinate)
                        lower[dim] -= 1
                        predecessors.append(
                            self._coordinate_to_index(tuple(lower))
                        )
                        predecessor_mask.append(1)
                    else:
                        predecessors.append(0)
                        predecessor_mask.append(0)
                lower_indices.append(predecessors)
                lower_masks.append(predecessor_mask)

            index_name = "level_{}_indices".format(level)
            lower_name = "level_{}_lower_indices".format(level)
            mask_name = "level_{}_lower_mask".format(level)
            self.register_buffer(index_name, th.LongTensor(indices))
            self.register_buffer(
                lower_name, th.LongTensor(lower_indices)
            )
            self.register_buffer(
                mask_name, th.ByteTensor(lower_masks)
            )
            self.level_buffer_names.append((index_name, lower_name, mask_name))

    def _coordinate_to_index(self, coordinate):
        index = 0
        for i, value in enumerate(coordinate):
            index += value * _product(self.lattice_sizes[i + 1 :])
        return index

    def _ordered_vertex_values(self, non_monotone_inputs):
        probabilities = self.auxiliary_network(non_monotone_inputs)
        batch_size = probabilities.size(0)
        values = probabilities.new(batch_size, self.num_vertices).zero_()

        for index_name, lower_name, mask_name in self.level_buffer_names:
            indices = getattr(self, index_name)
            lower_indices = getattr(self, lower_name)
            lower_mask = getattr(self, mask_name)

            gathered = values[:, lower_indices.view(-1)]
            gathered = gathered.view(
                batch_size, indices.numel(), self.n_monotone
            )
            mask = lower_mask.unsqueeze(0).expand_as(gathered)
            gathered = gathered.masked_fill(mask.eq(0), float("-inf"))
            lower_bound = gathered.max(dim=-1)[0]
            has_lower = lower_mask.sum(dim=-1).gt(0)
            no_lower = has_lower.eq(0).unsqueeze(0).expand_as(lower_bound)
            lower_bound = lower_bound.masked_fill(no_lower, 0.0)

            vertex_probability = probabilities.index_select(1, indices)
            vertex_value = lower_bound + vertex_probability * (
                1.0 - lower_bound
            )
            scatter_indices = indices.unsqueeze(0).expand(batch_size, -1)
            values = values.scatter(1, scatter_indices, vertex_value)

        if self.diagnostics_enabled:
            self._last_probabilities = probabilities.detach()
            self._last_vertex_values = values.detach()
        return values

    def set_diagnostics_enabled(self, enabled):
        self.diagnostics_enabled = bool(enabled)
        self._last_vertex_values = None
        self._last_probabilities = None

    @staticmethod
    def _histogram_percentile(histogram, fraction):
        cumulative = histogram.cumsum(dim=0)
        threshold = fraction * cumulative[-1]
        indices = th.nonzero(cumulative >= threshold)
        if indices.numel() == 0:
            return histogram.new_tensor(0.0)
        return (indices[0, 0].float() + 0.5) / histogram.numel()

    def get_diagnostics(self, valid_rows, near_zero_threshold=1e-4):
        if self._last_vertex_values is None:
            return {}

        vertex_values = self._last_vertex_values[valid_rows]
        probabilities = self._last_probabilities[valid_rows]
        if vertex_values.numel() == 0:
            return {}

        diagnostics = {
            "hll_aux_probability_mean": probabilities.mean(),
            "hll_aux_probability_saturation_frac": (
                (probabilities < 0.05) | (probabilities > 0.95)
            ).float().mean(),
            "hll_vertex_range_mean": (
                vertex_values.max(dim=1)[0] - vertex_values.min(dim=1)[0]
            ).mean(),
        }

        histogram = vertex_values.new_zeros(100)
        delta_sum = vertex_values.new_tensor(0.0)
        delta_count = 0
        near_zero_count = vertex_values.new_tensor(0.0)
        delta_mins = []
        delta_maxes = []
        level_sums = [vertex_values.new_tensor(0.0) for _ in range(3)]
        level_counts = [0, 0, 0]

        edge_lower_indices = self.edge_lower_indices.to(vertex_values.device)
        edge_upper_indices = self.edge_upper_indices.to(vertex_values.device)
        edge_dimensions = self.edge_dimensions.to(vertex_values.device)
        edge_level_buckets = self.edge_level_buckets.to(vertex_values.device)

        for dim in range(self.n_monotone):
            dim_mask = edge_dimensions.eq(dim)
            lower_indices = edge_lower_indices[dim_mask]
            upper_indices = edge_upper_indices[dim_mask]
            deltas = vertex_values.index_select(
                1, upper_indices
            ) - vertex_values.index_select(1, lower_indices)
            flat_deltas = deltas.reshape(-1)

            diagnostics["hll_vertex_delta_dim_{}_mean".format(dim)] = (
                flat_deltas.mean()
            )
            diagnostics[
                "hll_vertex_delta_dim_{}_near_zero_frac".format(dim)
            ] = flat_deltas.lt(near_zero_threshold).float().mean()

            delta_sum = delta_sum + flat_deltas.sum()
            delta_count += flat_deltas.numel()
            near_zero_count = near_zero_count + flat_deltas.lt(
                near_zero_threshold
            ).float().sum()
            delta_mins.append(flat_deltas.min())
            delta_maxes.append(flat_deltas.max())
            histogram = histogram + th.histc(
                flat_deltas.detach().clamp(min=0.0, max=1.0).float(),
                bins=100,
                min=0.0,
                max=1.0,
            ).to(histogram.dtype)

            dim_levels = edge_level_buckets[dim_mask]
            for level in range(3):
                level_mask = dim_levels.eq(level)
                if level_mask.any():
                    level_deltas = deltas[:, level_mask]
                    level_sums[level] = level_sums[level] + level_deltas.sum()
                    level_counts[level] += level_deltas.numel()

        count = max(delta_count, 1)
        diagnostics.update(
            {
                "hll_vertex_delta_mean": delta_sum / count,
                "hll_vertex_delta_p10": self._histogram_percentile(
                    histogram, 0.1
                ),
                "hll_vertex_delta_p90": self._histogram_percentile(
                    histogram, 0.9
                ),
                "hll_vertex_delta_min": th.stack(delta_mins).min(),
                "hll_vertex_delta_max": th.stack(delta_maxes).max(),
                "hll_vertex_delta_near_zero_frac": near_zero_count / count,
            }
        )
        level_names = ("low", "mid", "high")
        for level, name in enumerate(level_names):
            if level_counts[level] > 0:
                diagnostics[
                    "hll_vertex_delta_{}_level_mean".format(name)
                ] = level_sums[level] / level_counts[level]
        return diagnostics

    def _interpolate_binary(self, monotone_inputs, vertex_values):
        weights = monotone_inputs.new(monotone_inputs.size(0), 1).fill_(1.0)
        for dim in range(self.n_monotone):
            coordinate = monotone_inputs[:, dim]
            dim_weights = th.stack((1.0 - coordinate, coordinate), dim=1)
            weights = (
                weights.unsqueeze(2) * dim_weights.unsqueeze(1)
            ).view(monotone_inputs.size(0), -1)
        return (vertex_values * weights).sum(dim=1, keepdim=True)

    def _interpolate_general(self, monotone_inputs, vertex_values):
        scaled = monotone_inputs * (self.mesh_size.float() - 1.0)
        lower = th.floor(scaled).long()
        for dim, size in enumerate(self.lattice_sizes):
            lower[:, dim] = th.clamp(lower[:, dim], min=0, max=size - 2)
        fraction = scaled - lower.float()

        corner_coordinates = lower.unsqueeze(1) + self.corners.unsqueeze(0)
        corner_indices = (
            corner_coordinates * self.index_coefficients
        ).sum(dim=-1)

        corner_mask = self.corners.float().unsqueeze(0)
        fraction = fraction.unsqueeze(1)
        corner_weights = (
            corner_mask * fraction + (1.0 - corner_mask) * (1.0 - fraction)
        ).prod(dim=-1)
        corner_values = th.gather(vertex_values, 1, corner_indices)
        return (corner_values * corner_weights).sum(dim=1, keepdim=True)

    def forward(self, monotone_inputs, non_monotone_inputs):
        vertex_values = self._ordered_vertex_values(non_monotone_inputs)
        if self.binary_lattice:
            return self._interpolate_binary(monotone_inputs, vertex_values)
        return self._interpolate_general(monotone_inputs, vertex_values)


class HLLMonotoneMixer(nn.Module):
    """HLL partial-monotone mixer replacing QMIX's hypernetwork mixer."""

    def __init__(self, args):
        super(HLLMonotoneMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.q_groups = _map_override(args, "hll_q_groups", self.n_agents)
        if self.q_groups is None:
            self.q_groups = self.n_agents
        self.q_groups = int(self.q_groups)
        if self.q_groups < 1 or self.q_groups > self.n_agents:
            raise ValueError(
                "hll_q_groups must be between 1 and n_agents"
            )
        self.q_grouping = _map_override(args, "hll_q_grouping", "sorted")
        if self.q_grouping not in ("sorted", "contiguous"):
            raise ValueError(
                "hll_q_grouping must be 'sorted' or 'contiguous'"
            )

        self.lattice_sizes = _lattice_sizes(
            _map_override(args, "hll_lattice_size", 2), self.q_groups
        )
        self.max_vertices = _map_override(args, "hll_max_vertices", 4096)
        self.q_temperature = _map_override(args, "hll_q_temperature", 1.0)
        self.q_softsign_scale = _map_override(
            args, "hll_q_softsign_scale", 2.0
        )
        configured_coordinate_fn = getattr(
            args, "hll_q_coordinate_fn", None
        )
        if configured_coordinate_fn is None:
            configured_coordinate_fn = (
                "learned_sigmoid"
                if getattr(args, "hll_q_calibrator_enabled", False)
                else "sigmoid"
            )
        self.q_coordinate_fn = str(configured_coordinate_fn).lower()
        if self.q_coordinate_fn not in (
            "sigmoid",
            "learned_sigmoid",
            "softsign",
        ):
            raise ValueError(
                "hll_q_coordinate_fn must be 'sigmoid', "
                "'learned_sigmoid', or 'softsign'"
            )
        self.q_calibrator_enabled = (
            self.q_coordinate_fn == "learned_sigmoid"
        )
        self.q_residual_scale = _map_override(
            args, "hll_q_residual_scale", 0.0
        )
        self.q_residual_mode = _map_override(
            args, "hll_q_residual_mode", "sum"
        )

        self.aux_hidden_dim = getattr(args, "hll_aux_hidden_dim", 64)
        self.aux_depth = getattr(args, "hll_aux_depth", 2)
        self.aux_activation = getattr(args, "hll_aux_activation", "relu")
        self.aux_output_bias_init = getattr(
            args, "hll_aux_output_bias_init", -2.0
        )

        self.scale_hidden_dim = getattr(args, "hll_scale_hidden_dim", 32)
        self.scale_activation = getattr(args, "hll_scale_activation", "relu")
        self.min_output_scale = getattr(args, "hll_min_output_scale", 1e-3)
        self.state_value_dim = getattr(args, "hll_state_value_dim", 32)
        self.state_value_activation = getattr(
            args, "hll_state_value_activation", "relu"
        )
        self.diagnostics_enabled = False
        self._last_diagnostic_tensors = None

        if self.q_temperature <= 0:
            raise ValueError("hll_q_temperature must be positive")
        if self.q_softsign_scale <= 0:
            raise ValueError("hll_q_softsign_scale must be positive")
        self.q_calibrator = None
        if self.q_calibrator_enabled:
            self.q_calibrator = LearnedMonotoneQCalibrator(
                init_shift=getattr(
                    args, "hll_q_calibrator_init_shift", 0.0
                ),
                init_scale=getattr(
                    args, "hll_q_calibrator_init_scale", 1.0
                ),
                min_scale=getattr(
                    args, "hll_q_calibrator_min_scale", 0.25
                ),
            )
        if self.q_residual_scale < 0:
            raise ValueError(
                "hll_q_residual_scale must be non-negative to preserve IGM"
            )
        if self.q_residual_mode not in ("sum", "mean"):
            raise ValueError("hll_q_residual_mode must be 'sum' or 'mean'")
        if self.min_output_scale <= 0:
            raise ValueError("hll_min_output_scale must be positive")

        num_vertices = _product(self.lattice_sizes)
        auxiliary_network = HLLAuxiliaryNetwork(
            input_dim=self.state_dim,
            output_dim=num_vertices,
            hidden_dim=self.aux_hidden_dim,
            depth=self.aux_depth,
            activation=self.aux_activation,
            output_bias_init=self.aux_output_bias_init,
        )
        self.hll = HLLHierarchicalLattice(
            lattice_sizes=self.lattice_sizes,
            auxiliary_network=auxiliary_network,
            max_vertices=self.max_vertices,
        )

        self.output_scale = nn.Sequential(
            nn.Linear(self.state_dim, self.scale_hidden_dim),
            _activation(self.scale_activation),
            nn.Linear(self.scale_hidden_dim, 1),
        )
        self.state_value = StateValueNetwork(
            self.state_dim,
            hidden_dim=self.state_value_dim,
            activation=self.state_value_activation,
        )

    def set_diagnostics_enabled(self, enabled):
        self.diagnostics_enabled = bool(enabled)
        self._last_diagnostic_tensors = None
        self.hll.set_diagnostics_enabled(self.diagnostics_enabled)

    @staticmethod
    def _percentile(values, fraction):
        sorted_values = values.reshape(-1).sort()[0]
        index = int(round(fraction * (sorted_values.numel() - 1)))
        return sorted_values[index]

    def get_diagnostics(self, mask=None):
        tensors = self._last_diagnostic_tensors
        if tensors is None:
            return {}

        reference = tensors["output_scale"]
        if mask is None:
            valid_rows = reference.new_ones(reference.shape).gt(0).reshape(-1)
        else:
            valid_rows = mask.detach().reshape(-1).gt(0)
        if not valid_rows.any():
            return {}

        output_scale = reference.reshape(-1)[valid_rows]
        coordinates = tensors["q_coordinates"].reshape(
            -1, self.q_groups
        )[valid_rows]
        raw_qs = tensors["raw_qs"].reshape(
            -1, self.q_groups
        )[valid_rows]
        coordinate_derivative = tensors["coordinate_derivative"].reshape(
            -1, self.q_groups
        )[valid_rows]
        calibrator_shift = tensors["calibrator_shift"]
        calibrator_scale = tensors["calibrator_scale"]
        mixing_output = tensors["mixing_output"].reshape(-1)[valid_rows]
        state_value = tensors["state_value"].reshape(-1)[valid_rows]
        lattice_centered = tensors["lattice_centered"].reshape(-1)[valid_rows]
        eps = 1e-8
        mixing_rms = mixing_output.pow(2).mean().sqrt()
        value_rms = state_value.pow(2).mean().sqrt()
        low_saturation = coordinates < 0.05
        high_saturation = coordinates > 0.95

        diagnostics = {
            "hll_output_scale_mean": output_scale.mean(),
            "hll_output_scale_p10": self._percentile(output_scale, 0.1),
            "hll_output_scale_p90": self._percentile(output_scale, 0.9),
            "hll_output_scale_min": output_scale.min(),
            "hll_output_scale_max": output_scale.max(),
            "hll_q_coordinate_mean": coordinates.mean(),
            "hll_q_coordinate_p10": self._percentile(coordinates, 0.1),
            "hll_q_coordinate_p90": self._percentile(coordinates, 0.9),
            "hll_raw_q_mean": raw_qs.mean(),
            "hll_raw_q_std": raw_qs.std(unbiased=False),
            "hll_raw_q_p10": self._percentile(raw_qs, 0.1),
            "hll_raw_q_median": self._percentile(raw_qs, 0.5),
            "hll_raw_q_p90": self._percentile(raw_qs, 0.9),
            "hll_raw_q_min": raw_qs.min(),
            "hll_raw_q_max": raw_qs.max(),
            "hll_q_low_saturation_frac": low_saturation.float().mean(),
            "hll_q_high_saturation_frac": high_saturation.float().mean(),
            "hll_q_saturation_frac": (
                low_saturation | high_saturation
            ).float().mean(),
            "hll_coordinate_derivative_mean": coordinate_derivative.mean(),
            "hll_coordinate_derivative_p10": self._percentile(
                coordinate_derivative, 0.1
            ),
            # Retained as aliases so older result-processing scripts still run.
            "hll_sigmoid_sensitivity_mean": coordinate_derivative.mean(),
            "hll_sigmoid_sensitivity_p10": self._percentile(
                coordinate_derivative, 0.1
            ),
            "hll_q_calibrator_shift": calibrator_shift,
            "hll_q_calibrator_scale": calibrator_scale,
            "hll_lattice_centered_rms": lattice_centered.pow(2).mean().sqrt(),
            "hll_mixing_output_rms": mixing_rms,
            "hll_state_value_rms": value_rms,
            "hll_v_to_m_ratio": value_rms / (mixing_rms + eps),
        }
        coordinate_kind = (
            "agent" if self.q_groups == self.n_agents else "group"
        )
        for index in range(self.q_groups):
            coordinate = coordinates[:, index]
            coordinate_low = low_saturation[:, index]
            coordinate_high = high_saturation[:, index]
            prefix = "hll_q_{}_{}".format(coordinate_kind, index)
            diagnostics.update(
                {
                    "{}_coordinate_mean".format(prefix): coordinate.mean(),
                    "{}_raw_mean".format(prefix): raw_qs[:, index].mean(),
                    "{}_raw_std".format(prefix): raw_qs[:, index].std(
                        unbiased=False
                    ),
                    "{}_low_saturation_frac".format(prefix): (
                        coordinate_low.float().mean()
                    ),
                    "{}_high_saturation_frac".format(prefix): (
                        coordinate_high.float().mean()
                    ),
                    "{}_saturation_frac".format(prefix): (
                        (coordinate_low | coordinate_high).float().mean()
                    ),
                    "{}_coordinate_derivative_mean".format(prefix): (
                        coordinate_derivative[:, index].mean()
                    ),
                    "{}_sigmoid_sensitivity_mean".format(prefix): (
                        coordinate_derivative[:, index].mean()
                    ),
                }
            )
        diagnostics.update(
            self.hll.get_diagnostics(
                valid_rows,
                near_zero_threshold=getattr(
                    self.args, "hll_vertex_delta_near_zero_threshold", 1e-4
                ),
            )
        )
        return diagnostics

    def get_coordinate_credit_diagnostics(
        self, credit_grads, mask, near_zero_threshold=1e-4
    ):
        """Separate lattice-coordinate credit from coordinate attenuation."""
        tensors = self._last_diagnostic_tensors
        if (
            tensors is None
            or self.q_groups != self.n_agents
            or self.q_residual_scale != 0
        ):
            return {}

        derivative = tensors["coordinate_derivative"]
        if derivative.shape != credit_grads.shape:
            return {}
        valid_mask = mask.detach().expand_as(credit_grads).gt(0)
        if not valid_mask.any():
            return {}

        derivative = derivative.detach()
        coordinate_grads = credit_grads.detach() / derivative.clamp(min=1e-12)
        valid_derivative = derivative[valid_mask]
        valid_coordinate_grads = coordinate_grads[valid_mask]
        sorted_coordinate_grads = valid_coordinate_grads.sort()[0]

        def percentile(values, fraction):
            sorted_values = values.sort()[0]
            index = int(round(fraction * (sorted_values.numel() - 1)))
            return sorted_values[index]

        diagnostics = {
            "hll_coordinate_credit_grad_mean": valid_coordinate_grads.mean(),
            "hll_coordinate_credit_grad_p10": percentile(
                sorted_coordinate_grads, 0.1
            ),
            "hll_coordinate_credit_grad_median": percentile(
                sorted_coordinate_grads, 0.5
            ),
            "hll_coordinate_credit_grad_near_zero_frac": (
                valid_coordinate_grads.abs()
                .lt(near_zero_threshold)
                .float()
                .mean()
            ),
            "hll_calibration_attenuation_mean": valid_derivative.mean(),
            "hll_calibration_attenuation_p10": percentile(
                valid_derivative, 0.1
            ),
        }
        transition_mask = mask.detach().squeeze(-1).gt(0)
        valid_transitions = transition_mask.float().sum().clamp(min=1.0)
        for agent in range(self.n_agents):
            diagnostics[
                "hll_coordinate_credit_grad_agent_{}".format(agent)
            ] = (
                coordinate_grads[:, :, agent] * transition_mask.float()
            ).sum() / valid_transitions
            diagnostics[
                "hll_calibration_attenuation_agent_{}".format(agent)
            ] = (
                derivative[:, :, agent] * transition_mask.float()
            ).sum() / valid_transitions
        return diagnostics

    def _q_residual(self, agent_qs):
        if self.q_residual_mode == "mean":
            residual = agent_qs.mean(dim=1, keepdim=True)
        else:
            residual = agent_qs.sum(dim=1, keepdim=True)
        return self.q_residual_scale * residual

    def _group_q_inputs(self, agent_qs):
        if self.q_groups == self.n_agents:
            return agent_qs

        # Sorted quantile groups are permutation-invariant for homogeneous
        # teams; contiguous groups retain SMAC's type-sorted agent ordering for
        # heterogeneous teams. Both means are non-decreasing in every Q_i.
        if self.q_grouping == "sorted":
            grouped_source = th.sort(agent_qs, dim=1)[0]
        else:
            grouped_source = agent_qs
        grouped_qs = []
        for group in range(self.q_groups):
            start = group * self.n_agents // self.q_groups
            end = (group + 1) * self.n_agents // self.q_groups
            grouped_qs.append(grouped_source[:, start:end].mean(dim=1))
        return th.stack(grouped_qs, dim=1)

    def _coordinate_transform(self, lattice_qs):
        if self.q_coordinate_fn == "learned_sigmoid":
            coordinates, scale = self.q_calibrator(lattice_qs)
            shift = self.q_calibrator.shift
            derivative = coordinates * (1.0 - coordinates) / scale
        elif self.q_coordinate_fn == "softsign":
            shift = lattice_qs.new_zeros(())
            scale = lattice_qs.new_tensor(self.q_softsign_scale)
            coordinates = 0.5 * (1.0 + F.softsign(lattice_qs / scale))
            derivative = 0.5 / scale / (
                1.0 + lattice_qs.abs() / scale
            ).pow(2)
        else:
            shift = lattice_qs.new_zeros(())
            scale = lattice_qs.new_tensor(self.q_temperature)
            coordinates = th.sigmoid(lattice_qs / scale)
            derivative = coordinates * (1.0 - coordinates) / scale
        return coordinates, derivative, shift, scale

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, self.n_agents)

        lattice_qs = self._group_q_inputs(agent_qs)
        (
            q_coordinates,
            coordinate_derivative,
            calibrator_shift,
            calibrator_scale,
        ) = self._coordinate_transform(lattice_qs)
        lattice_output = self.hll(q_coordinates, states)
        output_scale = (
            F.softplus(self.output_scale(states)) + self.min_output_scale
        )
        q_residual = self._q_residual(agent_qs)
        state_value = self.state_value(states)
        lattice_centered = lattice_output - 0.5
        mixing_output = output_scale * lattice_centered
        q_tot = state_value + mixing_output + q_residual

        if self.diagnostics_enabled:
            steps = q_tot.size(0) // bs
            self._last_diagnostic_tensors = {
                "output_scale": output_scale.detach().view(bs, steps),
                "q_coordinates": q_coordinates.detach().view(
                    bs, steps, self.q_groups
                ),
                "raw_qs": lattice_qs.detach().view(
                    bs, steps, self.q_groups
                ),
                "coordinate_derivative": coordinate_derivative.detach().view(
                    bs, steps, self.q_groups
                ),
                "calibrator_shift": calibrator_shift.detach().clone(),
                "calibrator_scale": calibrator_scale.detach().clone(),
                "lattice_centered": lattice_centered.detach().view(bs, steps),
                "mixing_output": mixing_output.detach().view(bs, steps),
                "state_value": state_value.detach().view(bs, steps),
            }
        return q_tot.view(bs, -1, 1)
