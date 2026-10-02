REGISTRY = {}

from .basic_controller import BasicMAC
from .central_basic_controller import CentralBasicMAC
from .kaleidoscope_controller import KaleidoscopeMAC
from .s2q_controller import S2QMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["kaleidoscope_mac"] = KaleidoscopeMAC
REGISTRY["s2q_mac"] = S2QMAC

from .ices_controller import ICESMAC
REGISTRY["ices_mac"] = ICESMAC
