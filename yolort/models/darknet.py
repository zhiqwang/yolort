from .darknetv5 import (
    DarkNet,
    darknet_s_r3_1,
    darknet_m_r3_1,
    darknet_l_r3_1,
    darknet_s_r4_0,
    darknet_m_r4_0,
    darknet_l_r4_0,
    __all__ as dv5_all
)
from .darknetv6 import (
    DarkNetV6,
    darknet_s_r6_0,
    darknet_m_r6_0,
    darknet_l_r6_0,
    __all__ as dv6_all
)

__all__ = dv5_all + dv6_all
