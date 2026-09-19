from collections import deque
from statistics import mean

from .q_learner import QLearner


class KaleidoscopeQLearner(QLearner):
    """PyMARL QMIX learner with Kaleidoscope mask regularisation.

    The base learner keeps PyMARL's one-step target construction. This class
    only adds the official mask diversity objective and periodic inactive
    weight reinitialisation.
    """

    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        options = args.kaleidoscope_args
        self._td_history = deque(maxlen=int(options["deque_len"]))
        self._div_history = deque(maxlen=int(options["deque_len"]))
        self._div_coef = float(options["div_coef"])
        self._reset_interval = int(options["reset_interval"])
        self._reset_ratio = options["reset_ratio"]
        self._last_reset_t = 0

    def _before_train(self, t_env):
        self.mac.agent.set_mask_grad(True)
        t_max = int(getattr(self.args, "t_max", 0))
        if (
            self._reset_interval > 0
            and t_env - self._last_reset_t > self._reset_interval
            and (not t_max or t_max - t_env > self._reset_interval)
        ):
            self.mac.agent.reset_inactive_weights(self._reset_ratio)
            self._last_reset_t = t_env

    def _augment_loss(self, td_loss):
        div_loss = self.mac.agent.mask_diversity_loss()
        self._td_history.append(float(td_loss.detach().item()))
        self._div_history.append(float(div_loss.detach().item()))
        div_mean = mean(self._div_history)
        td_mean = mean(self._td_history)
        div_coef = abs(self._div_coef * td_mean / div_mean) if div_mean else self._div_coef
        return td_loss + div_coef * div_loss, {
            "loss_td": td_loss.detach(),
            "div_loss": div_loss.detach(),
            "div_coef": div_coef,
        }
