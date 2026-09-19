from modules.mixers.amco_monotone import AMCOMonotoneMixer
from modules.mixers.hll_monotone import HLLMonotoneMixer
from modules.mixers.lmn_monotone import LMNMonotoneMixer
from modules.mixers.monokan_monotone import MonoKANMonotoneMixer
from modules.mixers.qmix import QMixer
from modules.mixers.smm_monotone import SMMMonotoneMixer
from modules.mixers.smnn_monotone import SMNNMonotoneMixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qplex import DMAQer
from modules.mixers.kaleidoscope_pmix import REGISTRY as KALEIDOSCOPE_PMIX_REGISTRY


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
REGISTRY.update(KALEIDOSCOPE_PMIX_REGISTRY)
