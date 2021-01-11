from torch import nn

from .common import Conv
from .yolo import yolov5, create_model, yolov5s, yolov5m, yolov5l  # noqa

from utils.activations import Hardswish


def yolov5_onnx(
    cfg_path='yolov5s.yaml',
    pretrained=False,
    progress=True,
    num_classes=80,
    **kwargs,
):

    model = create_model(cfg_path=cfg_path, pretrained=pretrained, progress=progress,
                         num_classes=num_classes, **kwargs)
    for m in model.modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, Conv) and isinstance(m.act, nn.Hardswish):
            m.act = Hardswish()  # assign activation

    return model
