# Copyright (c) 2021, yolort team. All rights reserved.
import contextlib
import sys
from pathlib import Path

import torch
from torch import nn

from .models.yolo import Detect, Model
from .utils import attempt_download

__all__ = ["add_yolov5_context", "load_yolov5_model", "get_yolov5_size"]


@contextlib.contextmanager
def add_yolov5_context():
    """
    Temporarily add yolov5 folder to `sys.path`. Adapted from
    https://github.com/fcakyon/yolov5-pip/blob/0d03de6/yolov5/utils/general.py#L739-L754

    torch.hub handles it in the same way:
    https://github.com/pytorch/pytorch/blob/d3e36fa/torch/hub.py#L387-L416
    """
    path_ultralytics_yolov5 = str(Path(__file__).parent.resolve())
    try:
        sys.path.insert(0, path_ultralytics_yolov5)
        yield
    finally:
        sys.path.remove(path_ultralytics_yolov5)


def get_yolov5_size(depth_multiple, width_multiple):
    if depth_multiple == 0.33 and width_multiple == 0.25:
        return "n"
    if depth_multiple == 0.33 and width_multiple == 0.5:
        return "s"
    if depth_multiple == 0.67 and width_multiple == 0.75:
        return "m"
    if depth_multiple == 1.0 and width_multiple == 1.0:
        return "l"
    if depth_multiple == 1.33 and width_multiple == 1.25:
        return "x"
    raise NotImplementedError(
        f"Currently does't support architecture with depth: {depth_multiple} "
        f"and {width_multiple}, fell free to create a ticket labeled enhancement to us"
    )


def load_yolov5_model(checkpoint_path: str, inplace: bool = True, fuse: bool = True):
    """
    Creates a specified YOLOv5 model

    Args:
        checkpoint_path (str): path of the YOLOv5 model, i.e. 'yolov5s.pt'
        inplace (bool): An in-place operation. Default: True
        fuse (bool): fuse model Conv2d() + BatchNorm2d() layers. Default: True

    Returns:
        YOLOv5 pytorch model
    """

    with add_yolov5_context():
        ckpt = torch.load(attempt_download(checkpoint_path), map_location=torch.device("cpu"))
        if fuse:
            model = ckpt["ema" if ckpt.get("ema") else "model"].float().fuse().eval()
        else:  # without layer fuse
            model = ckpt["ema" if ckpt.get("ema") else "model"].float().eval()

        # Compatibility updates
        for m in model.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
                m.inplace = inplace  # pytorch 1.7.0 compatibility
                if type(m) is Detect:
                    if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
                        delattr(m, "anchor_grid")
                        setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)

        return model
