"""PMIX backends for Kaleidoscope utility networks.

Kaleidoscope changes the per-agent utility network, not the value mixer.  The
named registry entries below keep that composition explicit while preserving
the exact AMCO (MLP), HLL (lattice), and MonoKAN mixer implementations used by
their standalone PMIX baselines.
"""

from modules.mixers.amco_monotone import AMCOMonotoneMixer
from modules.mixers.hll_monotone import HLLMonotoneMixer
from modules.mixers.monokan_monotone import MonoKANMonotoneMixer


REGISTRY = {
    "kaleidoscope_pmix_mlp": AMCOMonotoneMixer,
    "kaleidoscope_pmix_lattice": HLLMonotoneMixer,
    "kaleidoscope_pmix_kan": MonoKANMonotoneMixer,
}
