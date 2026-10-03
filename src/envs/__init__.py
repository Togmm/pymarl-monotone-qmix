from .multiagentenv import MultiAgentEnv
import sys
import os


def smac_fn(**kwargs) -> MultiAgentEnv:
    """Load SMACv1 only when it is the selected environment.

    SMACv1 and SMACv2 both bind pysc2 modules at import time.  Keeping this
    import lazy mirrors ePyMARL's dynamic registration and avoids importing
    both implementations in one process.
    """
    from smac.env import StarCraft2Env

    return StarCraft2Env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = smac_fn


def lbf_fn(**kwargs) -> MultiAgentEnv:
    """Construct the Gymnasium/Gym LBF adapter lazily.

    LBF is optional in the original PyMARL dependency set, so importing the
    adapter here would make ordinary SMAC runs fail before the environment is
    selected.
    """
    from .lbf_wrapper import LBFWrapper

    return LBFWrapper(**kwargs)


def smacv2_fn(**kwargs) -> MultiAgentEnv:
    """Construct the optional SMACv2 adapter lazily."""
    from .smacv2_wrapper import SMACv2Wrapper

    return SMACv2Wrapper(**kwargs)


REGISTRY["lbf"] = lbf_fn
REGISTRY["sc2v2"] = smacv2_fn

if sys.platform == "linux":
    # Resolve relative to the repository rather than the caller's working
    # directory. A conda environment may define an obsolete SC2PATH; retain
    # an explicit caller value, otherwise use this checkout's SC2 install.
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.environ.setdefault("SC2PATH", os.path.join(_repo_root, "3rdparty", "StarCraftII"))
