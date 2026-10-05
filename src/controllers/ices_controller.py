"""ICES controller with separate exploitation and privileged exploration MACs."""

import torch as th

from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY

from .basic_controller import BasicMAC


class ICESMAC(BasicMAC):
    """BasicMAC plus an intrinsic policy that can observe the global state."""

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self._uses_kaleidoscope = str(args.agent).startswith("kaleidoscope")
        if self._uses_kaleidoscope:
            self.id_source = getattr(args, "kaleidoscope_id_source", "agent_id")
            if self.id_source == "unit_type":
                self.n_masks = int(args.n_unit_types)
            elif self.id_source == "agent_id":
                self.n_masks = int(args.n_agents)
            else:
                raise ValueError("kaleidoscope_id_source must be 'unit_type' or 'agent_id'")
        state_dim = int(args.state_shape) if isinstance(args.state_shape, int) else int(th.tensor(args.state_shape).prod().item())
        int_input_shape = self._get_input_shape(scheme) + state_dim
        intrinsic_agent = getattr(args, "ices_intrinsic_agent", "rnn")
        self.int_agent = agent_REGISTRY[intrinsic_agent](
            int_input_shape, args
        )
        self.int_hidden_states = None
        # The ICES selector consumes both policies; the regular selector is
        # still available for evaluation and for callers using ``forward``.
        self.action_selector = action_REGISTRY[getattr(args, "action_selector", "epsilon_expl")](args)

    def _build_kaleidoscope_inputs(self, batch, t):
        bs = batch.batch_size
        observations = batch["obs"][:, t]
        if self.id_source == "unit_type":
            type_onehot = observations[..., -self.n_masks :]
            type_sum = type_onehot.sum(dim=-1)
            valid = th.logical_or(type_sum.eq(0), type_sum.eq(1))
            if not bool(valid.all()):
                raise ValueError("Kaleidoscope unit-type masks require a one-hot observation suffix")
            mask_ids = type_onehot.argmax(dim=-1).long()
        else:
            mask_ids = th.arange(self.n_agents, device=batch.device).view(1, -1).expand(bs, -1)
        inputs = [observations]
        if self.args.obs_last_action:
            inputs.append(
                th.zeros_like(batch["actions_onehot"][:, t])
                if t == 0 else batch["actions_onehot"][:, t - 1]
            )
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        inputs = th.cat([item.reshape(bs, self.n_agents, -1) for item in inputs], dim=-1)
        return inputs.reshape(bs * self.n_agents, -1), mask_ids.reshape(-1)

    def forward(self, ep_batch, t, test_mode=False):
        if not self._uses_kaleidoscope:
            return super().forward(ep_batch, t, test_mode=test_mode)
        agent_inputs, mask_ids = self._build_kaleidoscope_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, mask_ids)
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                reshaped = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def _build_int_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = [batch["obs"][:, t], batch["state"][:, t].unsqueeze(1).expand(-1, self.n_agents, -1)]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)
            )
        return th.cat([item.reshape(bs, self.n_agents, -1) for item in inputs], dim=-1).reshape(
            bs * self.n_agents, -1
        )

    def int_forward(self, ep_batch, t, test_mode=False):
        inputs = self._build_int_inputs(ep_batch, t)
        outputs, self.int_hidden_states = self.int_agent(inputs, self.int_hidden_states)
        return outputs.view(ep_batch.batch_size, self.n_agents, -1)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode)
        avail = ep_batch["avail_actions"][:, t_ep]
        ratio = 0.0 if test_mode else float(getattr(self.args, "ices_int_ratio", 0.1))
        if not test_mode:
            finish = float(getattr(self.args, "ices_int_finish", ratio))
            t_max = max(float(getattr(self.args, "t_max", 1)), 1.0)
            ratio = max(ratio * (1.0 - float(t_env) / t_max), finish)
        # Evaluation and decentralized deployment must not require the
        # privileged global state.  The selector never uses the intrinsic
        # branch when ratio is zero, so avoid constructing it in that case.
        int_qvals = (
            th.zeros_like(qvals)
            if ratio <= 0.0
            else self.int_forward(ep_batch, t_ep, test_mode=test_mode)
        )
        selected = self.action_selector.select_action(
            qvals[bs], int_qvals[bs], avail[bs], t_env, ratio, test_mode=test_mode
        )
        if isinstance(selected, tuple):
            return selected[0]
        return selected

    def init_hidden(self, batch_size):
        super().init_hidden(batch_size)
        self.int_hidden_states = self.int_agent.init_hidden().unsqueeze(0).expand(
            batch_size, self.n_agents, -1
        )

    def parameters(self):
        return list(self.agent.parameters()) + list(self.int_agent.parameters())

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.int_agent.load_state_dict(other_mac.int_agent.state_dict())

    def to(self, device):
        self.agent.to(device)
        self.int_agent.to(device)
        return self

    def cuda(self):
        return self.to(th.device(getattr(self.args, "device", "cuda")))

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.int_agent.state_dict(), "{}/int_agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location="cpu"))
        self.int_agent.load_state_dict(th.load("{}/int_agent.th".format(path), map_location="cpu"))
