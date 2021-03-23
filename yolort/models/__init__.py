# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from torch import nn

from .common import Conv
from .yolo_module import YOLOModule

from ..utils.activations import Hardswish, SiLU

from typing import Any


def yolov5s(upstream_version: str = 'r4.0', export_friendly: bool = False, **kwargs: Any):
    """
    Args:
        upstream_version (str): model released by the upstream YOLOv5. Possible values
            are 'r3.1' and 'r4.0'. Default: 'r4.0'.
        export_friendly (bool): Deciding whether to use (ONNX/TVM) export friendly mode.
            Default: False.
    """
    if upstream_version == 'r3.1':
        model = YOLOModule(arch="yolov5_darknet_pan_s_r31", **kwargs)
    elif upstream_version == 'r4.0':
        model = YOLOModule(arch="yolov5_darknet_pan_s_r40", **kwargs)
    else:
        raise NotImplementedError("Currently only supports r3.1 and r4.0 versions")

    if export_friendly:
        _export_module_friendly(model)

    return model


def yolov5m(upstream_version: str = 'r4.0', export_friendly: bool = False, **kwargs: Any):
    """
    Args:
        upstream_version (str): model released by the upstream YOLOv5. Possible values
            are 'r3.1' and 'r4.0'. Default: 'r4.0'.
        export_friendly (bool): Deciding whether to use (ONNX/TVM) export friendly mode.
            Default: False.
    """
    if upstream_version == 'r3.1':
        model = YOLOModule(arch="yolov5_darknet_pan_m_r31", **kwargs)
    elif upstream_version == 'r4.0':
        model = YOLOModule(arch="yolov5_darknet_pan_m_r40", **kwargs)
    else:
        raise NotImplementedError("Currently only supports r3.1 and r4.0 versions")

    if export_friendly:
        _export_module_friendly(model)

    return model


def yolov5l(upstream_version: str = 'r4.0', export_friendly: bool = False, **kwargs: Any):
    """
    Args:
        upstream_version (str): model released by the upstream YOLOv5. Possible values
            are 'r3.1' and 'r4.0'. Default: 'r4.0'.
        export_friendly (bool): Deciding whether to use (ONNX/TVM) export friendly mode.
            Default: False.
    """
    if upstream_version == 'r3.1':
        model = YOLOModule(arch="yolov5_darknet_pan_l_r31", **kwargs)
    elif upstream_version == 'r4.0':
        model = YOLOModule(arch="yolov5_darknet_pan_l_r40", **kwargs)
    else:
        raise NotImplementedError("Currently only supports r3.1 and r4.0 versions")

    if export_friendly:
        _export_module_friendly(model)

    return model


def yolotr(upstream_version: str = 'r4.0', export_friendly: bool = False, **kwargs: Any):
    """
    Args:
        upstream_version (str): model released by the upstream YOLOv5. Possible values
            are 'r3.1' and 'r4.0'. Default: 'r4.0'.
        export_friendly (bool): Deciding whether to use (ONNX/TVM) export friendly mode.
            Default: False.
    """
    if upstream_version == 'r4.0':
        model = YOLOModule(arch="yolov5_darknet_tan_s_r40", **kwargs)
    else:
        raise NotImplementedError("Currently only supports r4.0 versions")

    if export_friendly:
        _export_module_friendly(model)

    return model


def _export_module_friendly(model):
    for m in model.modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, Conv):
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()  # assign activation
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()
