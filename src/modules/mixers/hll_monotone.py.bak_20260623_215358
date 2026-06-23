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

        return values

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

        self.lattice_sizes = _lattice_sizes(
            getattr(args, "hll_lattice_size", 2), self.n_agents
        )
        self.max_vertices = getattr(args, "hll_max_vertices", 4096)
        self.q_temperature = getattr(args, "hll_q_temperature", 1.0)

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

        if self.q_temperature <= 0:
            raise ValueError("hll_q_temperature must be positive")
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

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, self.n_agents)

        q_coordinates = th.sigmoid(agent_qs / self.q_temperature)
        lattice_output = self.hll(q_coordinates, states)
        output_scale = (
            F.softplus(self.output_scale(states)) + self.min_output_scale
        )
        q_tot = self.state_value(states) + output_scale * (
            lattice_output - 0.5
        )
        # q_tot = output_scale * (lattice_output - 0.5)
        return q_tot.view(bs, -1, 1)
