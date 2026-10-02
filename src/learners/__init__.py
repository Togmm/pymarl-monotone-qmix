from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .wqmix_learner import WQMixLearner
from .qplex_learner import QPLEXLearner
from .kaleidoscope_learner import KaleidoscopeQLearner
from .s2q_learner import S2QLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["wqmix_learner"] = WQMixLearner
REGISTRY["qplex_learner"] = QPLEXLearner
REGISTRY["kaleidoscope_q_learner"] = KaleidoscopeQLearner
REGISTRY["s2q_learner"] = S2QLearner

from .ices_learner import ICESQLearner
REGISTRY["ices_q_learner"] = ICESQLearner
