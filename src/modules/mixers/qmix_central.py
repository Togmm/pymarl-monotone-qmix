import numpy as np
import torch as th
import torch.nn as nn


class QMixerCentralFF(nn.Module):
    """Official WQMIX centralized feed-forward mixer."""
    def __init__(self, args):
        super(QMixerCentralFF, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.central_mixing_embed_dim
        inp = self.state_dim + self.n_agents * args.central_action_embed
        self.net = nn.Sequential(nn.Linear(inp, self.embed_dim), nn.ReLU(),
                                 nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU(),
                                 nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU(),
                                 nn.Linear(self.embed_dim, 1))
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim), nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, self.n_agents * self.args.central_action_embed)
        out = self.net(th.cat([states, agent_qs], dim=1)) + self.V(states)
        return out.view(bs, -1, 1)
