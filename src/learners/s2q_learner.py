import copy

import torch as th
import torch.nn.functional as F
from torch.optim import Adam, RMSprop

from components.episode_buffer import EpisodeBatch
from controllers.s2q_controller import S2QCentralMAC
from modules.mixers import REGISTRY as mixer_REGISTRY
from modules.s2q import S2QCentralMixer, S2QEncoderDecoder
from utils.rl_utils import build_td_lambda_targets


class S2QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args, self.mac, self.logger = args, mac, logger
        self.n_branches = int(getattr(args, "s2q_k", getattr(args, "K", 2))) + 1
        self.s2q_alpha = float(getattr(args, "s2q_alpha", 1.0))
        self.last_target_update_episode = 0
        self.log_stats_t = -args.learner_log_interval - 1
        assert args.common_reward, "S2Q requires a common reward for value mixing"
        # Keep one mixer per S2Q branch, but allow its width to be reduced
        # independently from the repository QMIX mixer.  This preserves the
        # official three-mixer topology while making total S2Q capacity
        # comparable to the single-mixer baselines.
        mixer_args = copy.copy(args)
        if hasattr(args, "s2q_mixing_embed_dim"):
            mixer_args.mixing_embed_dim = args.s2q_mixing_embed_dim
        if hasattr(args, "s2q_hypernet_embed"):
            mixer_args.hypernet_embed = args.s2q_hypernet_embed
        self.mixers = th.nn.ModuleList([mixer_REGISTRY[args.mixer](mixer_args) for _ in range(self.n_branches)])
        self.central_mac = S2QCentralMAC(scheme, {}, args)
        self.target_central_mac = copy.deepcopy(self.central_mac)
        self.central_mixer = S2QCentralMixer(args)
        self.target_central_mixer = copy.deepcopy(self.central_mixer)
        self.target_mac = copy.deepcopy(mac)
        self.target_mixers = th.nn.ModuleList([copy.deepcopy(m) for m in self.mixers])
        hidden_dim = int(getattr(args, "s2q_hidden_dim", getattr(args, "rnn_hidden_dim", 64)))
        state_dim = int(th.tensor(args.state_shape).prod().item())
        self.encoder_decoder = S2QEncoderDecoder(hidden_dim * args.n_agents, state_dim, args)
        self.mac.set_encoder_decoder(self.encoder_decoder)
        self.params = list(mac.parameters()) + list(self.central_mac.parameters()) + list(self.central_mixer.parameters()) + list(self.encoder_decoder.parameters())
        for mixer in self.mixers:
            self.params += list(mixer.parameters())
        # Count only online trainable modules. Target copies are intentionally
        # excluded because they do not increase policy capacity.
        self.online_parameter_count = sum(parameter.numel() for parameter in self.params)
        self._logged_parameter_count = False
        optimizer = getattr(args, "optimizer", "rmsprop").lower()
        if optimizer == "adam":
            self.optimiser = Adam(self.params, lr=args.lr)
        else:
            self.optimiser = RMSprop(self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

    def _forward_mac(self, mac, batch):
        outputs = [[] for _ in range(self.n_branches)]
        histories = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            result = mac.forward(batch, t=t)
            for index in range(self.n_branches):
                outputs[index].append(result[index])
            histories.append(result[-1])
        return [th.stack(values, dim=1) for values in outputs], th.stack(histories, dim=1)

    @staticmethod
    def _masked_argmax(q_values, avail_actions):
        safe_avail_actions = avail_actions.float()
        no_valid_action = safe_avail_actions.sum(dim=-1, keepdim=True).le(0)
        if no_valid_action.any():
            safe_avail_actions = safe_avail_actions.clone()
            fallback = no_valid_action.squeeze(-1)
            safe_avail_actions[..., 0] = th.where(
                fallback,
                th.ones_like(safe_avail_actions[..., 0]),
                safe_avail_actions[..., 0],
            )
        masked = q_values.detach().clone()
        masked[safe_avail_actions == 0] = -float("inf")
        return masked.max(dim=-1, keepdim=True)[1]

    @staticmethod
    def _gather_actions(values, actions):
        index = actions.unsqueeze(-1).expand(*actions.shape, values.size(-1))
        return th.gather(values, 3, index).squeeze(3)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        self.mac.set_encoder_decoder(self.encoder_decoder)
        online_heads, histories = self._forward_mac(self.mac, batch)
        chosen = [
            mixer(
                th.gather(q[:, :-1], 3, actions).squeeze(3),
                batch["state"][:, :-1],
            )
            for mixer, q in zip(self.mixers, online_heads)
        ]
        self.central_mac.init_hidden(batch.batch_size)
        central_online = th.stack([self.central_mac.forward(batch, t=t) for t in range(batch.max_seq_length)], dim=1)
        central_chosen = self.central_mixer(self._gather_actions(central_online[:, :-1], actions), batch["state"][:, :-1])

        with th.no_grad():
            current_actions = [self._masked_argmax(q, avail_actions) for q in online_heads]
            self.target_central_mac.init_hidden(batch.batch_size)
            target_central = th.stack([self.target_central_mac.forward(batch, t=t) for t in range(batch.max_seq_length)], dim=1)
            target_central_q = [self.target_central_mixer(self._gather_actions(target_central, action_index), batch["state"]) for action_index in current_actions]
            td_lambda = getattr(self.args, "td_lambda", None)
            if td_lambda is None:
                targets = rewards + self.args.gamma * (1 - terminated) * target_central_q[0][:, 1:]
            else:
                targets = build_td_lambda_targets(rewards, terminated, mask, target_central_q[0], self.args.n_agents, self.args.gamma, td_lambda)

        central_error = central_chosen - targets.detach()
        errors = [chosen[0] - targets.detach()]
        is_max = [(actions.squeeze(-1) == current[:, :-1].squeeze(-1)).all(dim=2) for current in current_actions]
        suppression = self.s2q_alpha * central_chosen.detach()
        errors.append(chosen[1] - (targets.detach() - suppression * is_max[0].unsqueeze(-1)))
        errors.append(chosen[2] - (targets.detach() - suppression * (is_max[0] | is_max[1]).unsqueeze(-1)))
        central_comp = [central_chosen.detach() >= target[:, :-1] for target in target_central_q]
        weight0 = th.where(central_comp[0] & central_comp[1] & central_comp[2], th.ones_like(central_error), th.full_like(central_error, float(self.args.w_c)))
        weights = [weight0, th.where(errors[1] < 0, th.ones_like(errors[1]), th.full_like(errors[1], float(self.args.w_c))), th.where(errors[2] < 0, th.ones_like(errors[2]), th.full_like(errors[2], float(self.args.w_c)))]
        head_mask = mask.expand_as(errors[0])
        denom = head_mask.sum().clamp(min=1.0)
        losses = [(weight.detach() * error.pow(2) * head_mask).sum() / denom for weight, error in zip(weights, errors)]
        central_loss = (central_error.pow(2) * mask).sum() / denom
        history_input = histories[:, :-1].reshape(batch.batch_size, histories.size(1) - 1, -1).detach()
        predicted_state, predicted_probs = self.encoder_decoder(history_input)
        stacked = th.stack([q[:, :-1] for q in target_central_q], dim=-1).squeeze(-2)
        target_probs = th.softmax(stacked / float(self.args.s2q_temperature), dim=-1)
        latent_loss = F.mse_loss(predicted_state, batch["state"][:, :-1]) + F.kl_div(
            th.log(predicted_probs.clamp_min(1e-8)),
            target_probs.detach(),
            reduction="batchmean",
        )
        loss = sum(losses) + central_loss + latent_loss
        self.optimiser.zero_grad(); loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets(); self.last_target_update_episode = episode_num
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            valid = head_mask.sum().item()
            if not self._logged_parameter_count:
                self.logger.log_stat("s2q_online_params", float(self.online_parameter_count), t_env)
                self._logged_parameter_count = True
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("loss_td", sum(losses).item(), t_env)
            self.logger.log_stat("loss_central", central_loss.item(), t_env)
            self.logger.log_stat("loss_latent", latent_loss.item(), t_env)
            self.logger.log_stat("grad_norm", float(grad_norm), t_env)
            self.logger.log_stat("td_error_abs", (errors[0].abs() * head_mask).sum().item() / valid, t_env)
            self.logger.log_stat("q_taken_mean", (chosen[0] * head_mask).sum().item() / valid, t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / mask.sum().clamp(min=1).item(), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        for target, online in zip(self.target_mixers, self.mixers):
            target.load_state_dict(online.state_dict())
        self.target_central_mac.load_state(self.central_mac)
        self.target_central_mixer.load_state_dict(self.central_mixer.state_dict())

    def cuda(self):
        self.mac.cuda(); self.target_mac.cuda(); self.mixers.cuda(); self.target_mixers.cuda()
        self.central_mac.cuda(); self.target_central_mac.cuda(); self.central_mixer.cuda(); self.target_central_mixer.cuda(); self.encoder_decoder.cuda()

    def save_models(self, path):
        self.mac.save_models(path); self.central_mac.save_models(path)
        for index, mixer in enumerate(self.mixers):
            th.save(mixer.state_dict(), "{}/mixer_{}.th".format(path, index))
        th.save(self.central_mixer.state_dict(), "{}/central_mixer.th".format(path))
        th.save(self.encoder_decoder.state_dict(), "{}/encoder_decoder.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path); self.central_mac.load_models(path)
        for index, mixer in enumerate(self.mixers):
            mixer.load_state_dict(th.load("{}/mixer_{}.th".format(path, index), map_location="cpu"))
        self.central_mixer.load_state_dict(th.load("{}/central_mixer.th".format(path), map_location="cpu"))
        self.encoder_decoder.load_state_dict(th.load("{}/encoder_decoder.th".format(path), map_location="cpu"))
        self.mac.set_encoder_decoder(self.encoder_decoder); self._update_targets()
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location="cpu"))
