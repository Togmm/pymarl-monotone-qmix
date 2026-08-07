import copy
import math
from components.episode_buffer import EpisodeBatch
from modules.mixers import REGISTRY as mixer_REGISTRY
import torch as th
from torch.optim import RMSprop


def _credit_diagnostics(credit_grads, mask, near_zero_threshold=1e-4):
    """Summarise local dQ_tot/dQ_i values over non-padded transitions."""
    valid_mask = mask.detach().expand_as(credit_grads).gt(0)
    valid_grads = credit_grads.detach()[valid_mask]
    if valid_grads.numel() == 0:
        return {}

    sorted_grads = valid_grads.sort()[0]

    def percentile(fraction):
        index = int(round(fraction * (sorted_grads.numel() - 1)))
        return sorted_grads[index]

    diagnostics = {
        "credit_grad_mean": valid_grads.mean(),
        "credit_grad_std": valid_grads.std(unbiased=False),
        "credit_grad_median": percentile(0.5),
        "credit_grad_p10": percentile(0.1),
        "credit_grad_p90": percentile(0.9),
        "credit_grad_min": sorted_grads[0],
        "credit_grad_max": sorted_grads[-1],
        "credit_grad_near_zero_frac": (
            valid_grads.abs().lt(near_zero_threshold).float().mean()
        ),
        "credit_grad_negative_frac": valid_grads.lt(-1e-7).float().mean(),
    }
    diagnostics["credit_grad_cv"] = diagnostics["credit_grad_std"] / (
        diagnostics["credit_grad_mean"].abs() + 1e-8
    )

    transition_mask = mask.detach().squeeze(-1).gt(0)
    nonnegative_grads = credit_grads.detach().clamp(min=0.0)
    grad_sums = nonnegative_grads.sum(dim=-1, keepdim=True)
    shares = nonnegative_grads / grad_sums.clamp(min=1e-8)
    if credit_grads.size(-1) > 1:
        entropy = -(
            shares * (shares + 1e-8).log()
        ).sum(dim=-1) / math.log(credit_grads.size(-1))
    else:
        entropy = th.ones_like(grad_sums.squeeze(-1))
    diagnostics["credit_entropy"] = entropy[transition_mask].mean()
    diagnostics["credit_max_share"] = shares.max(dim=-1)[0][
        transition_mask
    ].mean()

    valid_transitions = transition_mask.float().sum().clamp(min=1.0)
    for agent in range(credit_grads.size(-1)):
        diagnostics["credit_grad_agent_{}".format(agent)] = (
            credit_grads.detach()[:, :, agent] * transition_mask.float()
        ).sum() / valid_transitions
    return diagnostics


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer not in mixer_REGISTRY:
                raise ValueError(
                    "Mixer {} not recognised. Available mixers: {}".format(
                        args.mixer, ", ".join(sorted(mixer_REGISTRY.keys()))
                    )
                )
            self.mixer = mixer_REGISTRY[args.mixer](args)
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        should_log = t_env - self.log_stats_t >= self.args.learner_log_interval
        collect_mixer_diagnostics = (
            should_log
            and getattr(self.args, "mixer_diagnostics", False)
            and self.mixer is not None
            and hasattr(self.mixer, "get_diagnostics")
        )

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        individual_chosen_action_qvals = chosen_action_qvals

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            if hasattr(self.mixer, "set_train_step"):
                self.mixer.set_train_step(t_env)
            if hasattr(self.target_mixer, "set_train_step"):
                self.target_mixer.set_train_step(t_env)
            if hasattr(self.mixer, "set_diagnostics_enabled"):
                self.mixer.set_diagnostics_enabled(collect_mixer_diagnostics)
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        mixer_diagnostics = {}
        if collect_mixer_diagnostics:
            credit_grads = th.autograd.grad(
                outputs=chosen_action_qvals,
                inputs=individual_chosen_action_qvals,
                grad_outputs=mask,
                retain_graph=True,
                create_graph=False,
            )[0]
            mixer_diagnostics.update(
                _credit_diagnostics(
                    credit_grads,
                    mask,
                    near_zero_threshold=getattr(
                        self.args, "credit_near_zero_threshold", 1e-4
                    ),
                )
            )
            mixer_diagnostics.update(self.mixer.get_diagnostics(mask))
            if hasattr(self.mixer, "get_coordinate_credit_diagnostics"):
                mixer_diagnostics.update(
                    self.mixer.get_coordinate_credit_diagnostics(
                        credit_grads,
                        mask,
                        near_zero_threshold=getattr(
                            self.args, "credit_near_zero_threshold", 1e-4
                        ),
                    )
                )
            if hasattr(self.mixer, "get_state_credit_diagnostics"):
                mixer_diagnostics.update(
                    self.mixer.get_state_credit_diagnostics(
                        individual_chosen_action_qvals,
                        batch["state"][:, :-1],
                        credit_grads,
                        mask,
                    )
                )

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        if collect_mixer_diagnostics and hasattr(
            self.mixer, "get_gradient_diagnostics"
        ):
            mixer_diagnostics.update(self.mixer.get_gradient_diagnostics())
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if should_log:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            for key, value in mixer_diagnostics.items():
                self.logger.log_stat(key, value, t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
