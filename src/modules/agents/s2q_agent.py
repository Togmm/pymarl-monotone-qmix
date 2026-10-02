import torch as th
import torch.nn as nn
import torch.nn.functional as F


class S2QRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        hidden_dim = int(getattr(args, "s2q_hidden_dim", getattr(args, "rnn_hidden_dim", 64)))
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, args.n_actions)

    def init_hidden(self):
        return self.fc1.weight.new_zeros(1, self.hidden_dim)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h = self.rnn(x, hidden_state.reshape(-1, self.hidden_dim))
        return self.fc2(h), h


class S2QCentralAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.hidden_dim = int(getattr(args, "s2q_central_hidden_dim", 64))
        self.action_embed = int(getattr(args, "central_action_embed", 1))
        self.fc1 = nn.Linear(input_shape, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, args.n_actions * self.action_embed)

    def init_hidden(self):
        return self.fc1.weight.new_zeros(1, self.hidden_dim)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h = self.rnn(x, hidden_state.reshape(-1, self.hidden_dim))
        return self.fc2(h).reshape(-1, self.args.n_actions, self.action_embed), h
