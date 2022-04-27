# Copyright (c) 2021, yolort team. All rights reserved.

from typing import Any

from torch import nn

from .yolo import YOLO
from .yolov5 import YOLOv5

__all__ = [
    "YOLO",
    "YOLOv5",
    "yolov5n",
    "yolov5n6",
    "yolov5s",
    "yolov5s6",
    "yolov5m",
    "yolov5m6",
    "yolov5l",
    "yolov5ts",
]


def yolov5n(upstream_version: str = "r6.0", export_friendly: bool = False, **kwargs: Any):
    """
    Args:
        upstream_version (str): model released by the upstream YOLOv5. Possible values
            are ["r6.0"]. Default: "r6.0".
        export_friendly (bool): Deciding whether to use (ONNX/TVM) export friendly mode.
            Default: False.
    """
    if upstream_version == "r6.0":
        model = YOLOv5(arch="yolov5_darknet_pan_n_r60", **kwargs)
    else:
        raise NotImplementedError("Currently only supports r6.0 version")

    if export_friendly:
        _export_module_friendly(model)

    return model


def yolov5s(upstream_version: str = "r6.0", export_friendly: bool = False, **kwargs: Any):
    """
    Args:
        upstream_version (str): model released by the upstream YOLOv5. Possible values
            are ["r3.1", "r4.0", "r6.0"]. Default: "r6.0".
        export_friendly (bool): Deciding whether to use (ONNX/TVM) export friendly mode.
            Default: False.
    """
    if upstream_version == "r3.1":
        model = YOLOv5(arch="yolov5_darknet_pan_s_r31", **kwargs)
    elif upstream_version == "r4.0":
        model = YOLOv5(arch="yolov5_darknet_pan_s_r40", **kwargs)
    elif upstream_version == "r6.0":
        model = YOLOv5(arch="yolov5_darknet_pan_s_r60", **kwargs)
    else:
        raise NotImplementedError("Currently doesn't support this versions.")

    if export_friendly:
        _export_module_friendly(model)

    return model


def yolov5m(upstream_version: str = "r6.0", export_friendly: bool = False, **kwargs: Any):
    """
    Args:
        upstream_version (str): model released by the upstream YOLOv5. Possible values
            are ["r3.1", "r4.0", "r6.0"]. Default: "r6.0".
        export_friendly (bool): Deciding whether to use (ONNX/TVM) export friendly mode.
            Default: False.
    """
    if upstream_version == "r3.1":
        model = YOLOv5(arch="yolov5_darknet_pan_m_r31", **kwargs)
    elif upstream_version == "r4.0":
        model = YOLOv5(arch="yolov5_darknet_pan_m_r40", **kwargs)
    elif upstream_version == "r6.0":
        model = YOLOv5(arch="yolov5_darknet_pan_m_r60", **kwargs)
    else:
        raise NotImplementedError("Currently doesn't support this versions.")

    if export_friendly:
        _export_module_friendly(model)

    return model


def yolov5l(upstream_version: str = "r6.0", export_friendly: bool = False, **kwargs: Any):
    """
    Args:
        upstream_version (str): model released by the upstream YOLOv5. Possible values
            are ["r3.1", "r4.0", "r6.0"]. Default: "r6.0".
        export_friendly (bool): Deciding whether to use (ONNX/TVM) export friendly mode.
            Default: False.
    """
    if upstream_version == "r3.1":
        model = YOLOv5(arch="yolov5_darknet_pan_l_r31", **kwargs)
    elif upstream_version == "r4.0":
        model = YOLOv5(arch="yolov5_darknet_pan_l_r40", **kwargs)
    elif upstream_version == "r6.0":
        model = YOLOv5(arch="yolov5_darknet_pan_l_r60", **kwargs)
    else:
        raise NotImplementedError("Currently doesn't support this versions.")

    if export_friendly:
        _export_module_friendly(model)

    return model


def yolov5n6(upstream_version: str = "r6.0", export_friendly: bool = False, **kwargs: Any):
    """
    Args:
        upstream_version (str): model released by the upstream YOLOv5. Possible values
            are ["r6.0"]. Default: "r6.0".
        export_friendly (bool): Deciding whether to use (ONNX/TVM) export friendly mode.
            Default: False.
    """
    if upstream_version == "r6.0":
        model = YOLOv5(arch="yolov5_darknet_pan_n6_r60", size_divisible=64, **kwargs)
    else:
        raise NotImplementedError("Currently only supports r6.0 version")

    if export_friendly:
        _export_module_friendly(model)

    return model


def yolov5s6(upstream_version: str = "r6.0", export_friendly: bool = False, **kwargs: Any):
    """
    Args:
        upstream_version (str): model released by the upstream YOLOv5. Possible values
            are ["r6.0"]. Default: "r6.0".
        export_friendly (bool): Deciding whether to use (ONNX/TVM) export friendly mode.
            Default: False.
    """
    if upstream_version == "r6.0":
        model = YOLOv5(arch="yolov5_darknet_pan_s6_r60", size_divisible=64, **kwargs)
    else:
        raise NotImplementedError("Currently only supports r5.0 and r6.0 versions")

    if export_friendly:
        _export_module_friendly(model)

    return model


def yolov5m6(upstream_version: str = "r6.0", export_friendly: bool = False, **kwargs: Any):
    """
    Args:
        upstream_version (str): model released by the upstream YOLOv5. Possible values
            are ["r6.0"]. Default: "r6.0".
        export_friendly (bool): Deciding whether to use (ONNX/TVM) export friendly mode.
            Default: False.
    """
    if upstream_version == "r6.0":
        model = YOLOv5(arch="yolov5_darknet_pan_m6_r60", size_divisible=64, **kwargs)
    else:
        raise NotImplementedError("Currently only supports r5.0 and r6.0 versions")

    if export_friendly:
        _export_module_friendly(model)

    return model


def yolov5ts(upstream_version: str = "r4.0", export_friendly: bool = False, **kwargs: Any):
    """
    Args:
        upstream_version (str): model released by the upstream YOLOv5. Possible values
            are "r4.0". Default: "r4.0".
        export_friendly (bool): Deciding whether to use (ONNX/TVM) export friendly mode.
            Default: False.
    """
    if upstream_version == "r4.0":
        model = YOLOv5(arch="yolov5_darknet_tan_s_r40", **kwargs)
    else:
        raise NotImplementedError("Currently only supports r4.0 versions")

    if export_friendly:
        _export_module_friendly(model)

    return model


def _export_module_friendly(model):
    from yolort.v5 import Conv
    from yolort.v5.utils.activations import Hardswish, SiLU

    for m in model.modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, Conv):
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()  # assign activation
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()
