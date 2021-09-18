import contextlib
import sys
from pathlib import Path

import torch
from torch import nn

from yolort.models.common import Conv

from .models.yolo import Model, Detect
from .models.experimental import Ensemble
from .utils.downloads import attempt_download
from .utils.general import set_logging


@contextlib.contextmanager
def yolov5_in_syspath():
    """
    Temporarily add yolov5 folder to `sys.path`.

    torch.hub handles it in the same way:
    https://github.com/pytorch/pytorch/blob/75024e2/torch/hub.py#L387

    Proper fix for: #22, #134, #353, #1155, #1389, #1680, #2531, #3071
    No need for such workarounds: #869, #1052, #2949
    """
    yolov5_folder_dir = str(Path(__file__).parents[1].absolute())
    try:
        sys.path.insert(0, yolov5_folder_dir)
        yield
    finally:
        sys.path.remove(yolov5_folder_dir)


def load_model(model_path, device=None, autoshape=True, verbose=False):
    """
    Creates a specified YOLOv5 model

    Args:
        model_path (str): path of the model
        config_path (str): path of the config file
        device (str): select device that model will be loaded (cpu, cuda)
        pretrained (bool): load pretrained weights into the model
        autoshape (bool): make model ready for inference
        verbose (bool): if False, yolov5 logs will be silent

    Returns:
        pytorch model

    (Adapted from yolov5.hubconf.create)
    """
    # set logging
    set_logging(verbose=verbose)

    # set device if not given
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    attempt_download(model_path)  # download if not found locally
    with yolov5_in_syspath():
        model = torch.load(model_path, map_location=torch.device(device))
    if isinstance(model, dict):
        model = model["model"]  # load model
    hub_model = Model(model.yaml).to(next(model.parameters()).device)  # create
    hub_model.load_state_dict(model.float().state_dict())  # load state_dict
    hub_model.names = model.names  # class names
    model = hub_model

    if autoshape:
        model = model.autoshape()

    return model


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


class YOLOv5:
    def __init__(self, model_path, device=None, load_on_init=True):
        self.model_path = model_path
        self.device = device
        if load_on_init:
            Path(model_path).parents[0].mkdir(parents=True, exist_ok=True)
            self.model = load_model(model_path=model_path, device=device, autoshape=True)
        else:
            self.model = None

    def load_model(self):
        """
        Load yolov5 weight.
        """
        Path(self.model_path).parents[0].mkdir(parents=True, exist_ok=True)
        self.model = load_model(model_path=self.model_path, device=self.device, autoshape=True)

    def predict(self, image_list, size=640, augment=False):
        """
        Perform yolov5 prediction using loaded model weights.

        Returns results as a yolov5.models.common.Detections object.
        """
        assert self.model is not None, "before predict, you need to call .load_model()"
        results = self.model(imgs=image_list, size=size, augment=augment)
        return results
