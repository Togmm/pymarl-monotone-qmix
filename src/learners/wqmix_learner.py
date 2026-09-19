import copy
import torch as th
from torch.optim import RMSprop
from components.episode_buffer import EpisodeBatch
from modules.mixers import REGISTRY as MIXER_REGISTRY
from modules.mixers.qmix_central import QMixerCentralFF
from controllers import REGISTRY as mac_REGISTRY


class WQMixLearner:
    """Official OW-QMIX/CW-QMIX learner (weighted projection QMIX)."""
    def __init__(self, mac, scheme, logger, args):
        self.args, self.mac, self.logger = args, mac, logger
        # Keep the local branch pluggable: qmix reproduces the official
        # baseline, while a future PMIX-compatible mixer can be selected via
        # the same ``mixer`` config key.
        if args.mixer not in MIXER_REGISTRY:
            raise ValueError("Unknown local mixer '{}'".format(args.mixer))
        self.local_mixer = MIXER_REGISTRY[args.mixer](args)
        self.central_mixer = QMixerCentralFF(args)
        self.target_mac = copy.deepcopy(mac)
        self.target_local_mixer = copy.deepcopy(self.local_mixer)
        self.target_central_mixer = copy.deepcopy(self.central_mixer)
        self.central_mac = mac_REGISTRY[args.central_mac](scheme, None, args)
        self.target_central_mac = copy.deepcopy(self.central_mac)
        self.params = (list(mac.parameters()) + list(self.local_mixer.parameters()) +
                       list(self.central_mac.parameters()) + list(self.central_mixer.parameters()))
        self.optimiser = RMSprop(self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.last_target_update_episode = 0
        self.log_stats_t = -args.learner_log_interval - 1

    @staticmethod
    def _rollout(mac, batch):
        outs = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            outs.append(mac.forward(batch, t=t))
        return th.stack(outs, dim=1)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] *= (1 - terminated[:, :-1])
        avail = batch["avail_actions"]

        mac_out = self._rollout(self.mac, batch)
        chosen_agents = th.gather(mac_out[:, :-1], 3, actions).squeeze(3)
        central_out = self._rollout(self.central_mac, batch)
        idx = actions.unsqueeze(4).expand(-1, -1, -1, -1, self.args.central_action_embed)
        central_chosen_agents = th.gather(central_out[:, :-1], 3, idx).squeeze(3)

        target_out = self._rollout(self.target_mac, batch)
        target_out = target_out[:, 1:].masked_fill(avail[:, 1:] == 0, -9999999)
        live_out = mac_out.detach().masked_fill(avail == 0, -9999999)
        # Greedy actions at current transitions drive CW's projection test;
        # next-state greedy actions drive the double-Q target.
        cur_max_current = live_out[:, :-1].max(dim=3, keepdim=True)[1]
        if self.args.double_q:
            cur_max_actions = live_out[:, 1:].max(dim=3, keepdim=True)[1]
        else:
            cur_max_actions = target_out.max(dim=3, keepdim=True)[1]
        target_max_agents = th.gather(target_out, 3, cur_max_actions).squeeze(3)

        central_target_out = self._rollout(self.target_central_mac, batch)
        central_target_out = central_target_out.masked_fill(
            avail.unsqueeze(-1) == 0, -9999999)
        cidx = cur_max_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.args.central_action_embed)
        central_target_max_agents = th.gather(central_target_out[:, 1:], 3, cidx).squeeze(3)

        local_q = self.local_mixer(chosen_agents, batch["state"][:, :-1])
        central_q = self.central_mixer(central_chosen_agents, batch["state"][:, :-1])
        target_central_q = self.target_central_mixer(
            central_target_max_agents, batch["state"][:, 1:])
        targets = rewards + self.args.gamma * (1 - terminated) * target_central_q
        td_error = local_q - targets.detach()
        central_error = central_q - targets.detach()

        alpha = float(getattr(self.args, "w", 0.1))
        weights = th.full_like(td_error, alpha)
        if getattr(self.args, "hysteretic_qmix", True):  # OW-QMIX
            weights = th.where(td_error < 0, th.ones_like(weights), weights)
        else:  # CW-QMIX
            is_max = (actions == cur_max_current).min(dim=2)[0]
            max_qtot = self.target_central_mixer(
                th.gather(central_target_out[:, :-1], 3,
                          cur_max_current.unsqueeze(4).expand(-1, -1, -1, -1, self.args.central_action_embed)).squeeze(3),
                batch["state"][:, :-1])
            weights = th.where(is_max | (targets > max_qtot), th.ones_like(weights), weights)

        valid = mask.expand_as(td_error)
        denom = valid.sum().clamp(min=1.0)
        qmix_loss = (weights.detach() * td_error.pow(2) * valid).sum() / denom
        central_loss = (central_error.pow(2) * valid).sum() / denom
        loss = getattr(self.args, "qmix_loss", 1.0) * qmix_loss + getattr(self.args, "central_loss", 1.0) * central_loss

        self.optimiser.zero_grad(); loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets(); self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("qmix_loss", qmix_loss.item(), t_env)
            self.logger.log_stat("central_loss", central_loss.item(), t_env)
            self.logger.log_stat("w_to_use", (weights * valid).sum().item() / denom.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_local_mixer.load_state_dict(self.local_mixer.state_dict())
        self.target_central_mac.load_state(self.central_mac)
        self.target_central_mixer.load_state_dict(self.central_mixer.state_dict())

    def cuda(self):
        for x in (self.mac, self.target_mac, self.local_mixer, self.target_local_mixer,
                  self.central_mac, self.target_central_mac, self.central_mixer,
                  self.target_central_mixer): x.cuda()

    def save_models(self, path):
        self.mac.save_models(path); self.central_mac.save_models(path)
        th.save(self.local_mixer.state_dict(), path + "/local_mixer.th")
        th.save(self.central_mixer.state_dict(), path + "/central_mixer.th")
        th.save(self.optimiser.state_dict(), path + "/opt.th")

    def load_models(self, path):
        self.mac.load_models(path); self.target_mac.load_models(path)
        self.central_mac.load_models(path); self.target_central_mac.load_models(path)
        self.local_mixer.load_state_dict(th.load(path + "/local_mixer.th", map_location="cpu"))
        self.central_mixer.load_state_dict(th.load(path + "/central_mixer.th", map_location="cpu"))
        self._update_targets()
        self.optimiser.load_state_dict(th.load(path + "/opt.th", map_location="cpu"))
