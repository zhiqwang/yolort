import contextlib
import sys
from pathlib import Path

import torch
from torch import nn

from .models.common import Conv
from .models.yolo import Model, Detect
from .models.experimental import Ensemble
from .utils.downloads import attempt_download


@contextlib.contextmanager
def load_yolov5_model():
    """
    Temporarily add yolov5 folder to `sys.path`. Modified from:
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


def attempt_load(weights, map_location=None, inplace=True, fuse=True):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location=map_location)  # load
        if fuse:
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
        else:
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().eval())  # without layer fuse


    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = inplace  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        return model  # return ensemble
