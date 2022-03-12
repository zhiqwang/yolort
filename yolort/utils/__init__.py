# Copyright (c) 2020, yolort team. All rights reserved.

from typing import Callable, Dict, Mapping, Sequence, Union

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from .dependency import check_version
from .hooks import FeatureExtractor
from .image_utils import cv2_imshow, get_image_from_url, read_image_to_tensor
from .update_module_state import convert_yolov5_to_yolort, load_from_ultralytics
from .visualizer import Visualizer


__all__ = [
    "check_version",
    "cv2_imshow",
    "get_image_from_url",
    "get_callable_dict",
    "convert_yolov5_to_yolort",
    "load_from_ultralytics",
    "load_state_dict_from_url",
    "read_image_to_tensor",
    "FeatureExtractor",
    "Visualizer",
]


def get_callable_name(fn_or_class: Union[Callable, object]) -> str:
    return getattr(fn_or_class, "__name__", fn_or_class.__class__.__name__).lower()


def get_callable_dict(fn: Union[Callable, Mapping, Sequence]) -> Union[Dict, Mapping]:
    if isinstance(fn, Mapping):
        return fn
    elif isinstance(fn, Sequence):
        return {get_callable_name(f): f for f in fn}
    elif callable(fn):
        return {get_callable_name(fn): fn}
