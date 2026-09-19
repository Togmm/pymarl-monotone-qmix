from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

if sys.platform == "linux":
    # Resolve relative to the repository rather than the caller's working
    # directory. A conda environment may define an obsolete SC2PATH; retain
    # an explicit caller value, otherwise use this checkout's SC2 install.
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.environ.setdefault("SC2PATH", os.path.join(_repo_root, "3rdparty", "StarCraftII"))
