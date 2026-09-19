import math
from functools import partial

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class _CompareSTE(th.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        return F.relu(th.sign(inputs))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.tanh()


class KaleidoscopeLinear(nn.Linear):
    def __init__(self, in_features, out_features, n_masks, kalei_args, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.n_masks = n_masks
        self.reset_scale = kalei_args["threshold_reset_scale"]
        self.reset_bias = kalei_args["threshold_reset_bias"]
        thresholds = -(
            th.rand(n_masks, out_features, in_features)
            * kalei_args["threshold_init_scale"]
            + kalei_args["threshold_init_bias"]
        )
        self.sparse_thresholds = nn.Parameter(thresholds)

    def _ids(self, ids):
        ids = ids.reshape(-1).long().to(self.weight.device)
        if ids.numel() and (ids.min() < 0 or ids.max() >= self.n_masks):
            raise ValueError("Kaleidoscope mask id is outside configured range")
        return ids

    def _weights(self, ids):
        ids = self._ids(ids)
        threshold = th.sigmoid(self.sparse_thresholds)
        weights = th.sign(self.weight).unsqueeze(0) * F.relu(
            self.weight.abs().unsqueeze(0) - threshold
        )
        return weights[ids]

    def _weighted_masks(self):
        masks = _CompareSTE.apply(
            self.weight.detach().abs().unsqueeze(0) - th.sigmoid(self.sparse_thresholds)
        )
        return masks * self.weight.detach().abs().unsqueeze(0)

    def forward(self, inputs, ids):
        return th.bmm(self._weights(ids), inputs.unsqueeze(-1)).squeeze(-1) + self.bias

    def sparsities(self):
        with th.no_grad():
            weights = self._weights(th.arange(self.n_masks, device=self.weight.device))
            return 1.0 - weights.ne(0).float().flatten(1).mean(1), self.weight.numel()

    def reset_inactive(self, ratio):
        with th.no_grad():
            masks = _CompareSTE.apply(
                self.weight.detach().abs().unsqueeze(0) - th.sigmoid(self.sparse_thresholds)
            )
            inactive = masks.sum(0).eq(0)
            reset = inactive & th.rand_like(self.weight).lt(ratio)
            thresholds = -(
                th.rand_like(self.sparse_thresholds) * self.reset_scale + self.reset_bias
            )
            self.sparse_thresholds.copy_(th.where(reset.unsqueeze(0), thresholds, self.sparse_thresholds))
            weights = th.empty_like(self.weight)
            init.kaiming_uniform_(weights, a=math.sqrt(5))
            self.weight.copy_(th.where(reset, weights, self.weight))


class KaleidoscopeRNNAgent(nn.Module):
    """Official 1R3 Kaleidoscope agent using PyMARL's flat MAC inputs."""

    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.hidden_dim = args.rnn_hidden_dim
        self.n_masks = args.n_agents
        kalei_args = args.kaleidoscope_args
        make_linear = partial(
            KaleidoscopeLinear,
            n_masks=self.n_masks,
            kalei_args=kalei_args,
        )
        self.fc1 = make_linear(input_shape, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.fc2 = make_linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = make_linear(self.hidden_dim, self.hidden_dim)
        self.fc4 = make_linear(self.hidden_dim, args.n_actions)
        self.mask_layers = nn.ModuleList([self.fc1, self.fc2, self.fc3, self.fc4])
        self.reset_layers = [self.fc2, self.fc3, self.fc4]
        self.sparsity_layer_weights = kalei_args.get(
            "sparsity_layer_weights", [1.0, 2.0, 4.0, 8.0]
        )
        if len(self.sparsity_layer_weights) != 4:
            raise ValueError("sparsity_layer_weights must contain four values")
        self.set_mask_grad(False)

    def init_hidden(self):
        return self.fc1.weight.new_zeros(1, self.hidden_dim)

    def forward(self, inputs, hidden_state, mask_ids):
        inputs = inputs.reshape(-1, inputs.shape[-1])
        ids = mask_ids.reshape(-1)
        x = F.relu(self.fc1(inputs, ids))
        hidden = self.rnn(x, hidden_state.reshape(-1, self.hidden_dim))
        q = F.relu(self.fc2(hidden, ids))
        q = F.relu(self.fc3(q, ids))
        return self.fc4(q, ids), hidden

    @property
    def mask_parameters(self):
        return [layer.sparse_thresholds for layer in self.mask_layers]

    def set_mask_grad(self, enabled):
        for parameter in self.mask_parameters:
            parameter.requires_grad_(enabled)

    def mask_diversity_loss(self):
        loss = self.fc1.weight.new_zeros(())
        parameter_count = sum(layer.weight.numel() for layer in self.mask_layers)
        for layer, weight in zip(self.mask_layers, self.sparsity_layer_weights):
            masks = layer._weighted_masks()
            loss = loss + weight * (masks.unsqueeze(0) - masks.unsqueeze(1)).abs().sum()
        if self.n_masks < 2:
            return loss * 0.0
        return -loss / (self.n_masks * (self.n_masks - 1) * parameter_count)

    def reset_inactive_weights(self, ratio):
        ratios = [ratio] * len(self.reset_layers) if isinstance(ratio, (float, int)) else ratio
        for layer, current_ratio in zip(self.reset_layers, ratios):
            layer.reset_inactive(float(current_ratio))

    def get_sparsities(self):
        values, counts = zip(*(layer.sparsities() for layer in self.mask_layers))
        values = th.stack(values, dim=1)
        means = values.mean(0)
        variances = values.var(0, unbiased=False)
        total = sum(parameter.numel() for name, parameter in self.named_parameters() if "sparse_thresholds" not in name)
        overall = sum(count * mean for count, mean in zip(counts, means)) / total
        return means, variances, overall.item()
