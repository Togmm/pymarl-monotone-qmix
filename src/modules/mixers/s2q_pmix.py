"""PMIX mixer aliases for the S2Q baseline in PyMARL/SMACv1."""

from modules.mixers.amco_monotone import AMCOMonotoneMixer
from modules.mixers.hll_monotone import HLLMonotoneMixer
from modules.mixers.monokan_monotone import MonoKANMonotoneMixer


# S2Q creates one independent mixer per successive sub-value branch.  These
# names make that composition explicit while reusing the validated PMIX
# implementations used by the other PyMARL baselines.
REGISTRY = {
    "s2q_pmix_mlp": AMCOMonotoneMixer,
    "s2q_pmix_lattice": HLLMonotoneMixer,
    "s2q_pmix_kan": MonoKANMonotoneMixer,
}
