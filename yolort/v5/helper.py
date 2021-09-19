import contextlib
import sys
from pathlib import Path

import torch

from .models.yolo import Model
from .utils import attempt_download, set_logging

__all__ = ['add_yolov5_context', 'load_model']


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


def load_model(model_path: str, autoshape: bool = False, verbose: bool = True):
    """
    Creates a specified YOLOv5 model

    Args:
        model_path (str): path of the YOLOv5 model, i.e. 'yolov5s.pt'
        autoshape (bool): apply YOLOv5 .autoshape() wrapper to model. Default: False.
        verbose (bool): print all information to screen. Default: True.

    Returns:
        YOLOv5 pytorch model
    """
    set_logging(verbose=verbose)

    with add_yolov5_context():
        ckpt = torch.load(attempt_download(model_path), map_location=torch.device('cpu'))

    if isinstance(ckpt, dict):
        model_ckpt = ckpt["model"]  # load model

    model = Model(model_ckpt.yaml)  # create model
    model.load_state_dict(model_ckpt.float().state_dict())  # load state_dict

    if autoshape:
        model = model.autoshape()

    return model
