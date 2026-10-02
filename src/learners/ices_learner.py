"""ICES learner for EPyMARL's QMIX-compatible value factorisation API."""

import os
from collections import deque
from statistics import mean

import torch as th
from torch.optim import Adam, RMSprop

from modules.ices import ICESIntrinsicValue, ICESWorldModel

from .q_learner import QLearner


class ICESQLearner(QLearner):
    """Train exploitation Q-values and a privileged intrinsic policy jointly.

    The inherited TD path supplies double-Q, action masking, target networks,
    episode masks and any registered QMIX/PMIX mixer.  The auxiliary update is
    deliberately separate: world-model and intrinsic-policy parameters never
    receive gradients from the extrinsic TD loss.
    """

    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.n_agents = args.n_agents
        # QLearner sees all MAC parameters through ICESMAC.parameters().  Keep
        # the extrinsic optimiser restricted to the exploitation policy and
        # mixer, matching the reference ICES separation.
        self.params = list(mac.agent.parameters())
        if self.mixer is not None:
            self.params += list(self.mixer.parameters())
        # PyMARL's QMIX baseline uses RMSProp for the extrinsic TD path.  Keep
        # that optimizer and its defaults for fair comparisons; the ICES
        # auxiliary models below have their own Adam optimizers.
        self.optimiser = RMSprop(
            self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps
        )

        self.device = th.device(getattr(args, "device", "cuda" if args.use_cuda else "cpu"))
        state_dim = int(args.state_shape) if isinstance(args.state_shape, int) else int(th.tensor(args.state_shape).prod().item())
        self.world_model = ICESWorldModel(state_dim, args.n_agents, args.n_actions, args)
        world_lr = float(getattr(args, "ices_world_lr", 1e-4))
        self.world_optimiser = Adam(self.world_model.parameters(), lr=world_lr)

        int_input_dim = int(mac._get_input_shape(scheme)) + state_dim
        int_hidden = int(getattr(args, "ices_intrinsic_hidden_dim", getattr(args, "hidden_dim", 64)))
        self.int_critic = ICESIntrinsicValue(int_input_dim, int_hidden)
        int_lr = float(getattr(args, "ices_int_lr", 1e-2))
        int_c_lr = float(getattr(args, "ices_int_critic_lr", 1e-3))
        self.int_optimiser = Adam(mac.int_agent.parameters(), lr=int_lr)
        self.int_c_optimiser = Adam(self.int_critic.parameters(), lr=int_c_lr)

        self._td_history = deque(maxlen=100)
        self._div_history = deque(maxlen=100)
        kalei_args = getattr(args, "kaleidoscope_args", {}) or {}
        self._div_coef = float(kalei_args.get("div_coef", 5.0))
        self._reset_interval = int(kalei_args.get("reset_interval", 0))
        self._reset_ratio = kalei_args.get("reset_ratio", 0.2)
        self._last_reset_t = 0

        self._ices_batch = None

    def train(self, batch, t_env, episode_num):
        self._ices_batch = batch
        try:
            return super().train(batch, t_env, episode_num)
        finally:
            self._ices_batch = None

    def _before_train(self, t_env):
        if hasattr(self.mac.agent, "set_mask_grad"):
            self.mac.agent.set_mask_grad(True)
            t_max = int(getattr(self.args, "t_max", 0))
            if (
                self._reset_interval > 0
                and t_env - self._last_reset_t > self._reset_interval
                and (not t_max or t_max - t_env > self._reset_interval)
            ):
                self.mac.agent.reset_inactive_weights(self._reset_ratio)
                self._last_reset_t = t_env

    def _intrinsic_update(self, batch):
        states = batch["state"][:, :-1].float()
        next_states = batch["state"][:, 1:].float()
        actions = batch["actions"][:, :-1, :, 0].long()
        terminated = batch["terminated"][:, :-1].float()
        valid = batch["filled"][:, :-1].float()
        valid[:, 1:] = valid[:, 1:] * (1.0 - terminated[:, :-1])

        world_loss = self.world_model.loss(states, actions, next_states, valid)
        self.world_optimiser.zero_grad()
        world_loss.backward()
        world_grad = th.nn.utils.clip_grad_norm_(
            self.world_model.parameters(), self.args.grad_norm_clip
        )
        self.world_optimiser.step()
        with th.no_grad():
            intrinsic_rewards = self.world_model.intrinsic_rewards(states, actions)

        # Re-run the privileged policy over the batch.  The policy observes
        # global state only in this training branch; execution still uses the
        # ordinary observation-based exploitation policy.
        self.mac.init_hidden(batch.batch_size)
        logits, inputs = [], []
        for t in range(batch.max_seq_length - 1):
            inputs.append(self.mac._build_int_inputs(batch, t))
            logits.append(self.mac.int_forward(batch, t))
        logits = th.stack(logits, dim=1)
        int_inputs = th.stack(inputs, dim=0).view(
            batch.max_seq_length - 1, batch.batch_size, self.n_agents, -1
        ).permute(1, 0, 2, 3)
        avail = batch["avail_actions"][:, :-1].bool()
        masked_logits = logits.masked_fill(~avail, -1e9)
        log_probs = th.log_softmax(masked_logits, dim=-1)
        chosen_logp = th.gather(log_probs, -1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        values = self.int_critic(int_inputs.reshape(-1, int_inputs.shape[-1])).view(
            batch.batch_size, batch.max_seq_length - 1, self.n_agents
        )
        policy_mask = valid.expand_as(intrinsic_rewards)
        advantages = intrinsic_rewards - values.detach()
        denom = policy_mask.sum().clamp(min=1.0)
        entropy_coef = float(getattr(self.args, "ices_int_entropy_coef", 0.1))
        int_loss = -(
            advantages.detach() * chosen_logp + entropy_coef * entropy
        )
        int_loss = (int_loss * policy_mask).sum() / denom
        critic_loss = ((values - intrinsic_rewards) ** 2 * policy_mask).sum() / denom

        self.int_optimiser.zero_grad()
        int_loss.backward()
        int_grad = th.nn.utils.clip_grad_norm_(self.mac.int_agent.parameters(), self.args.grad_norm_clip)
        self.int_optimiser.step()
        self.int_c_optimiser.zero_grad()
        critic_loss.backward()
        critic_grad = th.nn.utils.clip_grad_norm_(self.int_critic.parameters(), self.args.grad_norm_clip)
        self.int_c_optimiser.step()
        return {
            "ices_world_loss": world_loss.detach(),
            "ices_world_grad_norm": world_grad.detach(),
            "ices_int_loss": int_loss.detach(),
            "ices_int_critic_loss": critic_loss.detach(),
            "ices_int_grad_norm": int_grad.detach(),
            "ices_int_critic_grad_norm": critic_grad.detach(),
            "ices_intrinsic_reward_mean": (intrinsic_rewards * policy_mask).sum() / denom,
        }

    def _augment_loss(self, td_loss):
        if self._ices_batch is None:
            return td_loss, {}
        stats = self._intrinsic_update(self._ices_batch)
        if not hasattr(self.mac.agent, "mask_diversity_loss"):
            return td_loss, stats
        div_loss = self.mac.agent.mask_diversity_loss()
        self._td_history.append(float(td_loss.detach().item()))
        self._div_history.append(float(div_loss.detach().item()))
        div_mean = mean(self._div_history)
        td_mean = mean(self._td_history)
        div_coef = abs(self._div_coef * td_mean / div_mean) if div_mean else self._div_coef
        stats.update({"loss_td": td_loss.detach(), "div_loss": div_loss.detach(), "div_coef": div_coef})
        return td_loss + div_coef * div_loss, stats

    def cuda(self):
        self.device = th.device(getattr(self.args, "device", "cuda"))
        self.mac.to(self.device)
        self.target_mac.to(self.device)
        if self.mixer is not None:
            self.mixer.to(self.device)
            self.target_mixer.to(self.device)
        self.world_model.to(self.device)
        self.int_critic.to(self.device)

    def save_models(self, path):
        super().save_models(path)
        th.save(self.world_model.state_dict(), os.path.join(path, "ices_world_model.th"))
        th.save(self.int_critic.state_dict(), os.path.join(path, "ices_int_critic.th"))
        th.save(self.world_optimiser.state_dict(), os.path.join(path, "ices_world_opt.th"))
        th.save(self.int_optimiser.state_dict(), os.path.join(path, "ices_int_opt.th"))
        th.save(self.int_c_optimiser.state_dict(), os.path.join(path, "ices_int_critic_opt.th"))

    def load_models(self, path):
        super().load_models(path)
        self.world_model.load_state_dict(th.load(os.path.join(path, "ices_world_model.th"), map_location="cpu"))
        self.int_critic.load_state_dict(th.load(os.path.join(path, "ices_int_critic.th"), map_location="cpu"))
        for optimizer, filename in (
            (self.world_optimiser, "ices_world_opt.th"),
            (self.int_optimiser, "ices_int_opt.th"),
            (self.int_c_optimiser, "ices_int_critic_opt.th"),
        ):
            full_path = os.path.join(path, filename)
            if os.path.exists(full_path):
                optimizer.load_state_dict(th.load(full_path, map_location="cpu"))
