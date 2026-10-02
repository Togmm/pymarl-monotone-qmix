from modules.mixers.amco_monotone import AMCOMonotoneMixer
from modules.mixers.pmix_monokan_ablation import (
    MonoKANQOnlyMixer,
    MonoKANQValueMixer,
)
from modules.mixers.hll_monotone import HLLMonotoneMixer
from modules.mixers.lmn_monotone import LMNMonotoneMixer
from modules.mixers.monokan_monotone import MonoKANMonotoneMixer
from modules.mixers.qmix import QMixer
from modules.mixers.smm_monotone import SMMMonotoneMixer
from modules.mixers.smnn_monotone import SMNNMonotoneMixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qplex import DMAQer
from modules.mixers.kaleidoscope_pmix import REGISTRY as KALEIDOSCOPE_PMIX_REGISTRY
from modules.mixers.s2q_pmix import REGISTRY as S2Q_PMIX_REGISTRY


REGISTRY = {}

REGISTRY["vdn"] = lambda args: VDNMixer()
REGISTRY["qmix"] = QMixer
REGISTRY["amco"] = AMCOMonotoneMixer
REGISTRY["hll"] = HLLMonotoneMixer
REGISTRY["lmn"] = LMNMonotoneMixer
REGISTRY["monokan"] = MonoKANMonotoneMixer
REGISTRY["smm"] = SMMMonotoneMixer
REGISTRY["smnn"] = SMNNMonotoneMixer
REGISTRY["dmaq"] = DMAQer
# PMIX-KAN state-path ablations: B1=M(q), B2=V(s)+M(q).
REGISTRY["monokan_b1"] = MonoKANQOnlyMixer
REGISTRY["monokan_b2"] = MonoKANQValueMixer
REGISTRY["pmix_monokan_b1"] = MonoKANQOnlyMixer
REGISTRY["pmix_monokan_b2"] = MonoKANQValueMixer
REGISTRY.update(KALEIDOSCOPE_PMIX_REGISTRY)
REGISTRY.update(S2Q_PMIX_REGISTRY)
