import contextlib
import sys
from pathlib import Path

import torch

from .models.yolo import Model
from .utils import attempt_download, set_logging

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
    raise NotImplementedError(
        f"Currently does't support architecture with depth: {depth_multiple} "
        f"and {width_multiple}, fell free to create a ticket labeled enhancement to us"
    )


def load_yolov5_model(checkpoint_path: str, autoshape: bool = False, verbose: bool = True):
    """
    Creates a specified YOLOv5 model

    Args:
        checkpoint_path (str): path of the YOLOv5 model, i.e. 'yolov5s.pt'
        autoshape (bool): apply YOLOv5 .autoshape() wrapper to model. Default: False.
        verbose (bool): print all information to screen. Default: True.

    Returns:
        YOLOv5 pytorch model
    """
    set_logging(verbose=verbose)

    with add_yolov5_context():
        ckpt = torch.load(attempt_download(checkpoint_path), map_location=torch.device("cpu"))

    if isinstance(ckpt, dict):
        model_ckpt = ckpt["model"]  # load model

    model = Model(model_ckpt.yaml)  # create model
    model.load_state_dict(model_ckpt.float().state_dict())  # load state_dict

    if autoshape:
        model = model.autoshape()

    return model
