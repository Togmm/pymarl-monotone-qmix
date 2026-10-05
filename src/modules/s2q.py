import numpy as np
import torch as th
import torch.nn as nn


class S2QEncoderDecoder(nn.Module):
    def __init__(self, input_dim, state_dim, args):
        super().__init__()
        hidden_dim = int(getattr(args, "s2q_encoder_hidden_dim", 64))
        latent_dim = int(getattr(args, "s2q_latent_dim", 32))
        n_branches = int(getattr(args, "s2q_k", getattr(args, "K", 2))) + 1
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.state_head = nn.Linear(hidden_dim, state_dim)
        self.prob_head = nn.Linear(hidden_dim, n_branches)

    def forward(self, histories):
        decoded = self.decoder(self.encoder(histories))
        return self.state_head(decoded), th.softmax(self.prob_head(decoded), dim=-1)


class S2QCentralMixer(nn.Module):
    """Unrestricted central mixer used by the S2Q target.

    The official S2Q implementation uses a feed-forward central mixer with a
    separate ``V(s)`` branch.  Keeping that branch is important: it lets the
    central critic represent a state-only baseline instead of forcing every
    state effect through the selected per-agent action values.  The embedding
    width is configurable so the complete S2Q model can be capacity-matched
    to the QMIX/PMIX baselines without removing any S2Q component.
    """

    def __init__(self, args):
        super().__init__()
        self.n_agents = int(args.n_agents)
        self.action_embed = int(getattr(args, "central_action_embed", 1))
        self.state_dim = int(np.prod(args.state_shape))
        hidden_dim = int(getattr(args, "central_mixing_embed_dim", 256))
        input_dim = self.n_agents * self.action_embed + self.state_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, agent_qs, states):
        batch, steps = agent_qs.shape[:2]
        q = agent_qs.reshape(batch, steps, -1)
        s = states.reshape(batch, steps, -1)
        return self.net(th.cat([q, s], dim=-1)) + self.V(s)
