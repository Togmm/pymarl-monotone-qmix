"""LBF adapter for the original PyMARL environment interface.

The current ePyMARL experiments use ``lbforaging`` through Gymma.  This
adapter intentionally uses the same common-reward and flattened-observation
protocol, while exposing the older PyMARL ``step -> (reward, done, info)``
API.  No LBF package is imported until this environment is selected.
"""

from collections.abc import Iterable

import numpy as np

from .multiagentenv import MultiAgentEnv


class LBFWrapper(MultiAgentEnv):
    def __init__(self, key, time_limit=50, seed=None, common_reward=True,
                 reward_scalarisation="sum", **kwargs):
        try:
            import gymnasium as gym
            from gymnasium.spaces import flatdim, flatten
            from gymnasium.wrappers import TimeLimit
        except ImportError:  # pragma: no cover - used by legacy PyMARL envs
            import gym
            from gym.spaces import flatdim, flatten
            from gym.wrappers import TimeLimit

        self._gym = gym
        self._flatdim = flatdim
        self._flatten = flatten
        self._env = gym.make(key, **kwargs)
        self._env = TimeLimit(self._env, max_episode_steps=int(time_limit))
        self._seed = seed
        self.common_reward = bool(common_reward)
        if not self.common_reward:
            raise ValueError("PyMARL LBF value mixing requires common_reward=True")
        if reward_scalarisation not in ("sum", "mean"):
            raise ValueError("reward_scalarisation must be 'sum' or 'mean'")
        self.reward_scalarisation = reward_scalarisation

        self.n_agents = int(self._env.unwrapped.n_agents)
        self._obs_space = self._env.observation_space
        self._action_space = self._env.action_space
        self._obs_dims = [self._space_dim(self._obs_space, i)
                          for i in range(self.n_agents)]
        self._action_dims = [self._action_n(self._action_space, i)
                             for i in range(self.n_agents)]
        self._obs_size = max(self._obs_dims)
        self.n_actions = max(self._action_dims)
        self.episode_limit = int(time_limit)
        self._obs = [np.zeros(self._obs_size, dtype=np.float32)
                     for _ in range(self.n_agents)]

    @staticmethod
    def _space_at(space, index):
        try:
            return space[index]
        except (IndexError, KeyError, TypeError):
            return space

    def _space_dim(self, space, index):
        return int(self._flatdim(self._space_at(space, index)))

    def _action_n(self, space, index):
        item = self._space_at(space, index)
        if hasattr(item, "n"):
            return int(item.n)
        if hasattr(item, "nvec"):
            # LBF uses one discrete action per agent.  Fail loudly for a
            # different action space instead of silently changing the task.
            nvec = np.asarray(item.nvec).reshape(-1)
            if len(nvec) != 1:
                raise ValueError("LBF adapter expects one discrete action per agent")
            return int(nvec[0])
        raise TypeError("Unsupported LBF action space: {}".format(type(item)))

    def _flatten_obs(self, obs):
        result = []
        for i in range(self.n_agents):
            item = obs[i]
            space = self._space_at(self._obs_space, i)
            value = np.asarray(self._flatten(space, item), dtype=np.float32).reshape(-1)
            if value.size < self._obs_size:
                value = np.pad(value, (0, self._obs_size - value.size))
            result.append(value[:self._obs_size])
        return result

    def step(self, actions):
        result = self._env.step([int(a) for a in actions])
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:  # Gym 0.21 compatibility
            obs, reward, terminated, info = result
            truncated = False
        self._obs = self._flatten_obs(obs)
        info = dict(info or {})
        if isinstance(truncated, Iterable):
            truncated = all(truncated)
        if isinstance(terminated, Iterable):
            terminated = all(terminated)
        if truncated:
            info["episode_limit"] = True
        if isinstance(reward, Iterable) and not isinstance(reward, (str, bytes)):
            values = [float(x) for x in reward]
            reward = sum(values)
            if self.reward_scalarisation == "mean" and values:
                reward /= len(values)
        else:
            reward = float(reward)
        return reward, bool(terminated or truncated), info

    def get_obs(self):
        return self._obs

    def get_obs_agent(self, agent_id):
        return self._obs[agent_id]

    def get_obs_size(self):
        return self._obs_size

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        return self.n_agents * self._obs_size

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        n = self._action_dims[agent_id]
        return [1] * n + [0] * (self.n_actions - n)

    def get_total_actions(self):
        return self.n_actions

    def reset(self):
        try:
            result = self._env.reset(seed=self._seed)
        except TypeError:  # Gym 0.21 compatibility
            if self._seed is not None and hasattr(self._env, "seed"):
                self._env.seed(self._seed)
            result = self._env.reset()
        # Gymnasium returns (observations, info), while older Gym returns
        # observations directly (which may itself be a tuple of agents).
        obs = result[0] if (isinstance(result, tuple) and len(result) == 2
                            and isinstance(result[1], dict)) else result
        self._obs = self._flatten_obs(obs)

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        self._seed = seed
        if hasattr(self._env, "seed"):
            self._env.seed(seed)
        return seed

    def save_replay(self):
        return None

    def get_stats(self):
        return {}
