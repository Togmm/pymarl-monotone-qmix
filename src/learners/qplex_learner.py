import copy
import torch as th
from torch.optim import RMSprop
from modules.mixers.qplex import DMAQer


class QPLEXLearner:
    """QPLEX learner following the official DMAQ implementation."""
    def __init__(self, mac, scheme, logger, args):
        self.args, self.mac, self.logger = args, mac, logger
        self.mixer = DMAQer(args); self.target_mixer = copy.deepcopy(self.mixer)
        self.target_mac = copy.deepcopy(mac)
        self.params = list(mac.parameters()) + list(self.mixer.parameters())
        self.optimiser = RMSprop(self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.last_target_update_episode = 0
        self.log_stats_t = -args.learner_log_interval - 1

    @staticmethod
    def _rollout(mac, batch):
        out = []; mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length): out.append(mac.forward(batch, t=t))
        return th.stack(out, dim=1)

    def train(self, batch, t_env, episode_num):
        rewards, actions = batch["reward"][:, :-1], batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float(); mask[:, 1:] *= (1 - terminated[:, :-1])
        avail = batch["avail_actions"]
        onehot = batch["actions_onehot"][:, :-1].float()
        mac_out = self._rollout(self.mac, batch)
        chosen = th.gather(mac_out[:, :-1], 3, actions).squeeze(3)
        live = mac_out.detach().masked_fill(avail == 0, -9999999)
        max_q_i, max_idx = live[:, :-1].max(dim=3)
        with th.no_grad():
            target_full = self._rollout(self.target_mac, batch)
            target = target_full[:, 1:].masked_fill(avail[:, 1:] == 0, -9999999)
            live_next = live[:, 1:]
            next_idx = live_next.max(dim=3, keepdim=True)[1] if self.args.double_q else target.max(dim=3, keepdim=True)[1]
            target_chosen = th.gather(target, 3, next_idx).squeeze(3)
            target_max_i = target.max(dim=3)[0]
            next_oh = th.zeros(target.size(0), target.size(1), target.size(2), self.args.n_actions,
                               device=target.device).scatter_(3, next_idx, 1.0)
        q_v = self.mixer(chosen, batch["state"][:, :-1], is_v=True)
        q_a = self.mixer(chosen, batch["state"][:, :-1], actions=onehot, max_q_i=max_q_i, is_v=False)
        q_tot = q_v + q_a
        with th.no_grad():
            tq_v = self.target_mixer(target_chosen, batch["state"][:, 1:], is_v=True)
            tq_a = self.target_mixer(target_chosen, batch["state"][:, 1:], actions=next_oh,
                                     max_q_i=target_max_i, is_v=False)
            target_tot = tq_v + tq_a
        targets = rewards + self.args.gamma * (1 - terminated) * target_tot
        td = q_tot - targets.detach(); valid = mask.expand_as(td)
        loss = 0.5 * (td.pow(2) * valid).sum() / valid.sum().clamp(min=1.0)
        self.optimiser.zero_grad(); loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip); self.optimiser.step()
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self.target_mac.load_state(self.mac); self.target_mixer.load_state_dict(self.mixer.state_dict())
            self.last_target_update_episode = episode_num
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("qplex_td_error_abs", (td.abs() * valid).sum().item() / valid.sum().item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env); self.log_stats_t = t_env

    def cuda(self):
        self.mac.cuda(); self.target_mac.cuda(); self.mixer.cuda(); self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path); th.save(self.mixer.state_dict(), path + "/mixer.th"); th.save(self.optimiser.state_dict(), path + "/opt.th")

    def load_models(self, path):
        self.mac.load_models(path); self.target_mac.load_models(path)
        self.mixer.load_state_dict(th.load(path + "/mixer.th", map_location="cpu")); self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optimiser.load_state_dict(th.load(path + "/opt.th", map_location="cpu"))
