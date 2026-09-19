import torch as th
from modules.agents import REGISTRY as agent_REGISTRY


class CentralBasicMAC:
    """Centralized controller used by the official OW/CW-QMIX branch."""
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.agent = agent_REGISTRY[args.central_agent](self._get_input_shape(scheme), args)
        self.agent_output_type = args.agent_output_type
        self.hidden_states = None

    def forward(self, ep_batch, t, test_mode=False):
        inputs = self._build_inputs(ep_batch, t)
        outs, self.hidden_states = self.agent(inputs, self.hidden_states)
        return outs.view(ep_batch.batch_size, self.n_agents, self.args.n_actions, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/central_agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/central_agent.th".format(path), map_location=lambda s, l: s))

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = [batch["obs"][:, t]]
        if getattr(self.args, "central_agent", "central_rnn") == "central_rnn_big":
            inputs[0] = batch["state"][:, t].unsqueeze(1).expand(-1, self.n_agents, -1)
        if self.args.obs_last_action:
            inputs.append(th.zeros_like(batch["actions_onehot"][:, t]) if t == 0 else batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        return th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if getattr(self.args, "central_agent", "central_rnn") == "central_rnn_big":
            input_shape = scheme["state"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
