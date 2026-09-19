import torch as th
import torch.nn as nn
import torch.nn.functional as F


class CentralRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CentralRNNAgent, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.central_rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.central_rnn_hidden_dim, args.central_rnn_hidden_dim)
        self.fc2 = nn.Linear(args.central_rnn_hidden_dim, args.n_actions * args.central_action_embed)

    def init_hidden(self):
        return self.fc1.weight.new_zeros(1, self.args.central_rnn_hidden_dim)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h = self.rnn(x, hidden_state.reshape(-1, self.args.central_rnn_hidden_dim))
        return self.fc2(h).reshape(-1, self.args.n_actions, self.args.central_action_embed), h
