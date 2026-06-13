from modules.mixers.amco_monotone import AMCOMonotoneMixer
from modules.mixers.qmix import QMixer
from modules.mixers.smm_monotone import SMMMonotoneMixer
from modules.mixers.smnn_monotone import SMNNMonotoneMixer
from modules.mixers.vdn import VDNMixer


REGISTRY = {}

REGISTRY["vdn"] = lambda args: VDNMixer()
REGISTRY["qmix"] = QMixer
REGISTRY["amco"] = AMCOMonotoneMixer
REGISTRY["smm"] = SMMMonotoneMixer
REGISTRY["smnn"] = SMNNMonotoneMixer
