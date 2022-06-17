# Copyright (c) 2021, yolort team. All rights reserved.

from .darknetv4 import (
    darknet_l_r3_1,
    darknet_l_r4_0,
    darknet_m_r3_1,
    darknet_m_r4_0,
    darknet_s_r3_1,
    darknet_s_r4_0,
    DarkNetV4,
)
from .darknetv6 import (
    darknet_l_r6_0,
    darknet_m_r6_0,
    darknet_n_r6_0,
    darknet_s_r6_0,
    darknet_x_r6_0,
    DarkNetV6,
)

__all__ = (
    "DarkNetV4",
    "DarkNetV6",
    "darknet_s_r3_1",
    "darknet_m_r3_1",
    "darknet_l_r3_1",
    "darknet_s_r4_0",
    "darknet_m_r4_0",
    "darknet_l_r4_0",
    "darknet_n_r6_0",
    "darknet_s_r6_0",
    "darknet_m_r6_0",
    "darknet_l_r6_0",
    "darknet_x_r6_0",
)
