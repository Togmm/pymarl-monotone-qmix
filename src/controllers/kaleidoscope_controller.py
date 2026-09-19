import torch as th

from .basic_controller import BasicMAC


class KaleidoscopeMAC(BasicMAC):
    """PyMARL MAC that supplies static agent-slot identities to the masks."""

    def forward(self, ep_batch, t, test_mode=False):
        inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        mask_ids = th.arange(self.n_agents, device=ep_batch.device).view(1, -1)
        mask_ids = mask_ids.expand(ep_batch.batch_size, -1).reshape(-1)
        agent_outs, self.hidden_states = self.agent(inputs, self.hidden_states, mask_ids)

        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                reshaped = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    @property
    def mask_parameters(self):
        return self.agent.mask_parameters

    @property
    def sparsities(self):
        return self.agent.get_sparsities()
