import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class S2QEpsilonGreedyActionSelector:
    def __init__(self, args):
        self.args = args
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time, decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.encoder_decoder = None
        self.strategy_selector = None

    def select_action(self, q_values, history, avail_actions, t_env, t_ep, test_mode=False):
        self.epsilon = self.schedule.eval(t_env)
        if test_mode:
            self.epsilon = getattr(self.args, "evaluation_epsilon", 0.0)

        # A finished or malformed environment can expose an all-zero action
        # mask. Categorical(avail_actions) would normalize that row by zero
        # and raise a Simplex/NaN error. Use action 0 as a defensive no-op
        # fallback for those rows while leaving valid masks unchanged.
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

        masked = [q.clone() for q in q_values]
        for q in masked:
            q[safe_avail_actions == 0] = -float("inf")
        greedy_actions = th.stack([q.max(dim=-1)[1] for q in masked], dim=-1)
        batch_size, n_agents = greedy_actions.shape[:2]
        if test_mode:
            selector = th.zeros(batch_size, dtype=th.long, device=history.device)
        else:
            if t_ep == 0:
                self.strategy_selector = "action_1" if th.rand(1).item() < 0.5 else "mixed"
            if self.strategy_selector == "mixed" and self.encoder_decoder is not None:
                with th.no_grad():
                    _, probs = self.encoder_decoder(history.reshape(batch_size, 1, -1))
                probs = th.nan_to_num(probs[:, -1], nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
                normalizer = probs.sum(dim=-1, keepdim=True)
                uniform = th.full_like(probs, 1.0 / probs.size(-1))
                probs = th.where(
                    normalizer > th.finfo(probs.dtype).eps,
                    probs / normalizer.clamp_min(th.finfo(probs.dtype).eps),
                    uniform,
                )
                selector = th.distributions.Categorical(probs=probs).sample()
            else:
                selector = th.zeros(batch_size, dtype=th.long, device=history.device)
        chosen = th.gather(greedy_actions, 2, selector.view(batch_size, 1, 1).expand(-1, n_agents, 1)).squeeze(-1)
        random_actions = th.distributions.Categorical(probs=safe_avail_actions).sample()
        random_mask = th.rand(batch_size, n_agents, device=history.device) < self.epsilon
        return th.where(random_mask, random_actions, chosen)


REGISTRY["s2q_epsilon_greedy"] = S2QEpsilonGreedyActionSelector

class EpsilonExplActionSelector:
    """Epsilon-greedy selector that occasionally follows ICES' policy."""

    def __init__(self, args):
        self.args = args
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time, decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, intrinsic_inputs, avail_actions, t_env, intrinsic_ratio, test_mode=False):
        self.epsilon = 0.0 if test_mode else self.schedule.eval(t_env)
        q_values = agent_inputs.clone()
        int_values = intrinsic_inputs.clone()
        q_values[avail_actions == 0] = -float("inf")
        int_values[avail_actions == 0] = -float("inf")
        int_probs = th.softmax(int_values, dim=-1)
        int_actions = th.distributions.Categorical(probs=int_probs).sample()
        random_actions = th.distributions.Categorical(avail_actions.float()).sample()
        random_pick = th.rand_like(q_values[..., 0]) < self.epsilon
        intrinsic_pick = th.rand_like(q_values[..., 0]) < float(intrinsic_ratio)
        behavior = th.where(intrinsic_pick, int_actions, q_values.max(dim=-1)[1])
        return th.where(random_pick, random_actions, behavior), th.distributions.Categorical(probs=int_probs).entropy()


REGISTRY["epsilon_expl"] = EpsilonExplActionSelector
