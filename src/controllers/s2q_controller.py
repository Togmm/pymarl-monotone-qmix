import torch as th

from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agents.s2q_agent import S2QCentralAgent, S2QRNNAgent


class S2QMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.n_branches = int(getattr(args, "s2q_k", getattr(args, "K", 2))) + 1
        input_shape = self._get_input_shape(scheme)
        self.agents = th.nn.ModuleList([S2QRNNAgent(input_shape, args) for _ in range(self.n_branches)])
        self.agent = self.agents[0]
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = [None] * self.n_branches

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        # ParallelRunner passes the indices of environments that are still
        # active. Slice before action selection so terminated environments,
        # whose action masks are often all zero, cannot produce invalid
        # categorical probabilities. The returned tensor is already in the
        # active-environment order expected by the runner.
        active_q_values = [q[bs] for q in outputs[: self.n_branches]]
        active_history = outputs[-1][bs]
        active_avail_actions = avail_actions[bs]
        return self.action_selector.select_action(
            active_q_values,
            active_history,
            active_avail_actions,
            t_env,
            t_ep,
            test_mode=test_mode,
        )

    def forward(self, ep_batch, t, test_mode=False):
        inputs = self._build_inputs(ep_batch, t)
        q_values, histories = [], []
        for index, agent in enumerate(self.agents):
            if test_mode:
                agent.eval()
            q, hidden = agent(inputs, self.hidden_states[index])
            self.hidden_states[index] = hidden
            q_values.append(q.view(ep_batch.batch_size, self.n_agents, -1))
            histories.append(hidden.view(ep_batch.batch_size, self.n_agents, -1))
        return (*q_values, histories[0])

    def set_encoder_decoder(self, encoder_decoder):
        self.action_selector.encoder_decoder = encoder_decoder

    def init_hidden(self, batch_size):
        self.hidden_states = [agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1) for agent in self.agents]

    def parameters(self):
        return self.agents.parameters()

    def load_state(self, other_mac):
        self.agents.load_state_dict(other_mac.agents.state_dict())

    def cuda(self):
        self.agents.cuda()

    def save_models(self, path):
        for index, agent in enumerate(self.agents):
            th.save(agent.state_dict(), "{}/agent_{}.th".format(path, index))

    def load_models(self, path):
        for index, agent in enumerate(self.agents):
            agent.load_state_dict(th.load("{}/agent_{}.th".format(path, index), map_location=lambda s, l: s))

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = [batch["obs"][:, t]]
        if self.args.obs_last_action:
            inputs.append(th.zeros_like(batch["actions_onehot"][:, t]) if t == 0 else batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        return th.cat([item.reshape(bs * self.n_agents, -1) for item in inputs], dim=-1)

    def _get_input_shape(self, scheme):
        shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            shape += self.n_agents
        return shape


class S2QCentralMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = int(scheme["state"]["vshape"])
        if getattr(args, "s2q_central_use_agent_id", True):
            input_shape += self.n_agents
        self.agent = S2QCentralAgent(input_shape, args)
        self.hidden_states = None

    def forward(self, ep_batch, t, test_mode=False):
        bs = ep_batch.batch_size
        state = ep_batch["state"][:, t].unsqueeze(1).expand(-1, self.n_agents, -1)
        inputs = [state]
        if getattr(self.args, "s2q_central_use_agent_id", True):
            inputs.append(th.eye(self.n_agents, device=ep_batch.device).unsqueeze(0).expand(bs, -1, -1))
        out, self.hidden_states = self.agent(th.cat(inputs, dim=-1).reshape(bs * self.n_agents, -1), self.hidden_states)
        return out.view(bs, self.n_agents, self.args.n_actions, -1)

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
