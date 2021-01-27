from torch import nn
from .common import Conv
from utils.activations import Hardswish

from .yolo import (yolov5_darknet_pan_s_r31 as yolov5s,
                   yolov5_darknet_pan_m_r31 as yolov5m,
                   yolov5_darknet_pan_l_r31 as yolov5l,
                   yolov5_darknet_pan_s_r40,
                   yolov5_darknet_pan_m_r40,
                   yolov5_darknet_pan_l_r40)


def yolov5_onnx(pretrained=False, progress=True, num_classes=80, **kwargs):

    model = yolov5s(pretrained=pretrained, progress=progress, num_classes=num_classes, **kwargs)
    for m in model.modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, Conv) and isinstance(m.act, nn.Hardswish):
            m.act = Hardswish()  # assign activation

    return model
