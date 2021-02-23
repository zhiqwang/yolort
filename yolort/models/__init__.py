# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from torch import nn

from .common import Conv
from .yolo_module import YOLOModule

from ..utils.activations import Hardswish

from typing import Any


def yolov5s(upstream_version: str ='v3.1', **kwargs: Any):
    if upstream_version == 'v3.1':
        model = YOLOModule(arch="yolov5_darknet_pan_s_r31", **kwargs)
    elif upstream_version == 'v4.0':
        model = YOLOModule(arch="yolov5_darknet_pan_s_r40", **kwargs)
    else:
        raise NotImplementedError("Currently only supports v3.1 and v4.0 versions")
    return model


def yolov5m(upstream_version: str ='v3.1', **kwargs: Any):
    if upstream_version == 'v3.1':
        model = YOLOModule(arch="yolov5_darknet_pan_m_r31", **kwargs)
    elif upstream_version == 'v4.0':
        model = YOLOModule(arch="yolov5_darknet_pan_m_r40", **kwargs)
    else:
        raise NotImplementedError("Currently only supports v3.1 and v4.0 versions")
    return model


def yolov5l(upstream_version: str ='v3.1', **kwargs: Any):
    if upstream_version == 'v3.1':
        model = YOLOModule(arch="yolov5_darknet_pan_l_r31", **kwargs)
    elif upstream_version == 'v4.0':
        model = YOLOModule(arch="yolov5_darknet_pan_l_r40", **kwargs)
    else:
        raise NotImplementedError("Currently only supports v3.1 and v4.0 versions")
    return model


def yolov5_onnx(pretrained=False, progress=True, num_classes=80, **kwargs):

    model = yolov5s(pretrained=pretrained, progress=progress, num_classes=num_classes, **kwargs)
    for m in model.modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, Conv) and isinstance(m.act, nn.Hardswish):
            m.act = Hardswish()  # assign activation

    return model
