"""Building blocks used by the ICES exploration scaffold.

The reference implementation uses two conditional VAEs.  This module keeps
the same observable mechanism (a global transition distribution contrasted
with an agent-wise leave-one-out distribution) while using a small diagonal
Gaussian predictor so that ICES works in the existing EPyMARL environments
without adding a Pyro dependency.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class GaussianTransitionModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, logvar_min=-4.0, logvar_max=2.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * output_dim),
        )
        self.output_dim = output_dim
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

    def forward(self, inputs):
        mean, logvar = self.net(inputs).split(self.output_dim, dim=-1)
        return mean, logvar.clamp(self.logvar_min, self.logvar_max)


class ICESWorldModel(nn.Module):
    """Global and leave-one-agent-out Gaussian transition predictors.

    ``intrinsic_rewards`` is the bounded KL divergence between the global
    posterior and each local posterior.  A local posterior sees the state and
    every action except the selected agent's action, which is the actionable
    contribution estimate used by ICES.
    """

    def __init__(self, state_dim, n_agents, n_actions, args):
        super().__init__()
        hidden = int(getattr(args, "ices_world_hidden_dim", getattr(args, "hidden_dim", 64)))
        action_dim = n_agents * n_actions
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.logvar_min = float(getattr(args, "ices_world_logvar_min", -4.0))
        self.logvar_max = float(getattr(args, "ices_world_logvar_max", 2.0))
        self.logvar_reg = float(getattr(args, "ices_world_logvar_reg", 1e-3))
        # The explicit missing-action mask is the one-hot equivalent of the
        # reference implementation's -1 sentinel.  It avoids conflating a
        # held-out action with an all-zero action block.
        model_input_dim = state_dim + action_dim + n_agents
        self.global_model = GaussianTransitionModel(
            model_input_dim, state_dim, hidden, self.logvar_min, self.logvar_max
        )
        self.local_model = GaussianTransitionModel(
            model_input_dim, state_dim, hidden, self.logvar_min, self.logvar_max
        )

    def _inputs(self, states, actions):
        actions_onehot = F.one_hot(actions.long(), self.n_actions).float()
        actions_onehot = actions_onehot.reshape(*actions.shape[:-1], -1)
        missing = th.zeros(*actions.shape[:-1], self.n_agents, device=states.device)
        return th.cat((states, actions_onehot, missing), dim=-1), actions_onehot

    @staticmethod
    def _nll(mean, logvar, target):
        return 0.5 * (logvar + (target - mean).pow(2) * th.exp(-logvar)).mean(dim=-1)

    def _distributions(self, states, actions):
        global_inputs, actions_onehot = self._inputs(states, actions)
        global_mean, global_logvar = self.global_model(global_inputs)
        local_means, local_logvars = [], []
        for agent in range(self.n_agents):
            local_actions = actions_onehot.clone()
            local_actions[..., agent * self.n_actions : (agent + 1) * self.n_actions] = 0.0
            missing = th.zeros(*actions.shape[:-1], self.n_agents, device=states.device)
            missing[..., agent] = 1.0
            local_inputs = th.cat((states, local_actions, missing), dim=-1)
            mean, logvar = self.local_model(local_inputs)
            local_means.append(mean)
            local_logvars.append(logvar)
        return (
            global_mean,
            global_logvar,
            th.stack(local_means, dim=-2),
            th.stack(local_logvars, dim=-2),
        )

    def intrinsic_rewards(self, states, actions):
        with th.no_grad():
            global_mean, global_logvar, local_mean, local_logvar = self._distributions(states, actions)
            global_mean = global_mean.unsqueeze(-2)
            global_logvar = global_logvar.unsqueeze(-2)
            global_var = global_logvar.exp()
            local_var = local_logvar.exp()
            kl = 0.5 * (
                local_logvar
                - global_logvar
                + (global_var + (global_mean - local_mean).pow(2)) / local_var
                - 1.0
            )
            # The paper bounds the surprise with 1-exp(-KL) before using it
            # as a policy-gradient reward.
            reward = 1.0 - th.exp(-kl.mean(dim=-1).clamp(min=0.0, max=20.0))
            return reward

    def loss(self, states, actions, next_states, valid):
        target = next_states - states
        global_mean, global_logvar, local_mean, local_logvar = self._distributions(states, actions)
        global_loss = self._nll(global_mean, global_logvar, target)
        local_loss = self._nll(
            local_mean,
            local_logvar,
            target.unsqueeze(-2).expand_as(local_mean),
        ).mean(dim=-1)
        transition_loss = global_loss + local_loss
        if self.logvar_reg:
            global_reg = global_logvar.square().mean(dim=-1)
            local_reg = local_logvar.square().mean(dim=(-1, -2))
            transition_loss = transition_loss + self.logvar_reg * (global_reg + local_reg)
        valid = valid.squeeze(-1).float()
        return (transition_loss * valid).sum() / valid.sum().clamp(min=1.0)


class ICESIntrinsicValue(nn.Module):
    """Recurrent per-agent baseline for the intrinsic policy-gradient objective."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim

    def init_hidden(self, batch_size, n_agents, device=None):
        device = device or self.fc1.weight.device
        return self.fc1.weight.new_zeros(batch_size, n_agents, self.hidden_dim, device=device)

    def forward(self, inputs, hidden_state):
        # inputs: [batch, agents, features], hidden_state: [batch, agents, hidden]
        x = F.relu(self.fc1(inputs))
        hidden = self.rnn(x.reshape(-1, self.hidden_dim), hidden_state.reshape(-1, self.hidden_dim))
        value = self.fc2(hidden).view(inputs.shape[0], inputs.shape[1], 1)
        return value, hidden.view(inputs.shape[0], inputs.shape[1], self.hidden_dim)
