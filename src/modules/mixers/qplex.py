import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class DMAQSIWeight(nn.Module):
    def __init__(self, args):
        super(DMAQSIWeight, self).__init__()
        self.n_agents, self.n_actions = args.n_agents, args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = self.n_agents * self.n_actions
        self.num_kernel = args.num_kernel
        e = args.adv_hypernet_embed
        self.key_extractors, self.agents_extractors, self.action_extractors = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        layers = getattr(args, "adv_hypernet_layers", 1)
        def net(inp, out):
            if layers == 1: return nn.Linear(inp, out)
            seq = [nn.Linear(inp, e), nn.ReLU()]
            for _ in range(layers - 2): seq += [nn.Linear(e, e), nn.ReLU()]
            seq += [nn.Linear(e, out)]
            return nn.Sequential(*seq)
        for _ in range(self.num_kernel):
            self.key_extractors.append(net(self.state_dim, 1))
            self.agents_extractors.append(net(self.state_dim, self.n_agents))
            self.action_extractors.append(net(self.state_dim + self.action_dim, self.n_agents))

    def forward(self, states, actions):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        data = th.cat([states, actions], dim=1)
        heads = []
        for k, a, x in zip(self.key_extractors, self.agents_extractors, self.action_extractors):
            key = th.abs(k(states)).repeat(1, self.n_agents) + 1e-10
            heads.append(key * th.sigmoid(a(states)) * th.sigmoid(x(data)))
        return th.stack(heads, dim=1).sum(dim=1)


class DMAQer(nn.Module):
    """QPLEX's duplex dueling multi-agent Q mixer."""
    def __init__(self, args):
        super(DMAQer, self).__init__()
        self.args, self.n_agents = args, args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        e = args.hypernet_embed
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, e), nn.ReLU(), nn.Linear(e, self.n_agents))
        self.V = nn.Sequential(nn.Linear(self.state_dim, e), nn.ReLU(), nn.Linear(e, self.n_agents))
        self.si_weight = DMAQSIWeight(args)

    def forward(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        qs = agent_qs.reshape(-1, self.n_agents)
        w = th.abs(self.hyper_w_final(states)) + 1e-10
        v = self.V(states)
        if getattr(self.args, "weighted_head", True):
            qs = w * qs + v
        if is_v:
            out = qs.sum(dim=1)
        else:
            max_q_i = max_q_i.reshape(-1, self.n_agents)
            if getattr(self.args, "weighted_head", True): max_q_i = w * max_q_i + v
            adv_q = (qs - max_q_i).detach()
            aw = self.si_weight(states, actions)
            out = (adv_q * (aw - 1.0 if getattr(self.args, "is_minus_one", True) else aw)).sum(dim=1)
        return out.view(bs, -1, 1)


QPLEXMixer = DMAQer
