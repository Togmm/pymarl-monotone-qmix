REGISTRY = {}

from .rnn_agent import RNNAgent
from .central_rnn_agent import CentralRNNAgent
from .kaleidoscope_agent import KaleidoscopeRNNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["kaleidoscope_rnn_1r3"] = KaleidoscopeRNNAgent
