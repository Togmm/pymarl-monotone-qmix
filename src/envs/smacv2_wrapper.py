"""SMACv2 adapter exposing the legacy PyMARL environment API."""

import random
from pathlib import Path

import numpy as np
import yaml

from .multiagentenv import MultiAgentEnv


CONFIG_DIR = Path(__file__).parent.parent / "config" / "envs" / "smacv2_configs"


def _seed_distributions(value, seed, visited=None):
    if visited is None:
        visited = set()
    if value is None or id(value) in visited:
        return
    visited.add(id(value))
    if hasattr(value, "rng"):
        value.rng = np.random.default_rng(seed)
        seed += 1
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (tuple, list, set)):
        children = value
    else:
        children = vars(value).values() if hasattr(value, "__dict__") else ()
    for child in children:
        _seed_distributions(child, seed, visited)
        seed += 1


def _load_scenario(map_name, seed=None, **overrides):
    from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

    path = CONFIG_DIR / (str(map_name) + ".yaml")
    if not path.is_file():
        raise FileNotFoundError("SMACv2 scenario config not found: {}".format(path))
    with path.open("r") as handle:
        config = yaml.safe_load(handle)
    env_args = dict(config["env_args"])
    env_args.update(overrides)
    env_args["seed"] = seed
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    env = StarCraftCapabilityEnvWrapper(**env_args)
    if seed is not None:
        _seed_distributions(env.env_key_to_distribution_map, seed)
    return env


class SMACv2Wrapper(MultiAgentEnv):
    def __init__(self, map_name, seed=None, **kwargs):
        self.map_name = map_name
        self._seed = seed
        self._kwargs = dict(kwargs)
        self.env = _load_scenario(map_name, seed=seed, **kwargs)
        self.episode_limit = self.env.episode_limit
        self.n_agents = self.env.get_env_info()["n_agents"]

    def step(self, actions):
        reward, terminated, info = self.env.step(actions)
        return float(reward), bool(terminated), dict(info or {})

    def get_obs(self):
        return self.env.get_obs()

    def get_obs_agent(self, agent_id):
        return self.env.get_obs_agent(agent_id)

    def get_obs_size(self):
        return self.env.get_obs_size()

    def get_state(self):
        return self.env.get_state()

    def get_state_size(self):
        return self.env.get_state_size()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        return self.env.get_avail_agent_actions(agent_id)

    def get_total_actions(self):
        return self.env.get_total_actions()

    def reset(self):
        result = None
        for _ in range(10):
            result = self.env.reset()
            if result is not None:
                break
        if result is None:
            raise RuntimeError("SMACv2 reset failed after 10 attempts")
        return result[0] if isinstance(result, tuple) else result

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        if seed is None:
            return self._seed
        self._seed = seed
        self.env.close()
        self.env = _load_scenario(self.map_name, seed=seed, **self._kwargs)
        self.episode_limit = self.env.episode_limit
        return seed

    def save_replay(self):
        return self.env.save_replay()

    def get_stats(self):
        return self.env.get_stats()

    def get_env_info(self):
        return self.env.get_env_info()
