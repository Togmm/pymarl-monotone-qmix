REGISTRY = {}

from .basic_controller import BasicMAC
from .central_basic_controller import CentralBasicMAC
from .kaleidoscope_controller import KaleidoscopeMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["kaleidoscope_mac"] = KaleidoscopeMAC
