# Copyright (c) 2020, yolort team. All rights reserved.

from typing import Any, Callable, Dict, Mapping, Sequence, Type, Union

from torch import Tensor
from .allreduce_norm import *
from .boxes import *
from .checkpoint import load_ckpt, save_checkpoint
from .compat import meshgrid
from .demo_utils import *
from .dist import *
from .ema import *
from .logger import WandbLogger, setup_logger
from .lr_scheduler import LRScheduler
from .metric import *
from .model_utils import *
from .visualize import *

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from .annotations_converter import AnnotationsConverter
from .dependency import check_version, is_module_available, requires_module
from .hooks import FeatureExtractor
from .image_utils import cv2_imshow, get_image_from_url, read_image_to_tensor
from .visualizer import Visualizer


__all__ = [
    "AnnotationsConverter",
    "FeatureExtractor",
    "Visualizer",
    "check_version",
    "contains_any_tensor",
    "cv2_imshow",
    "get_image_from_url",
    "get_callable_dict",
    "is_module_available",
    "load_state_dict_from_url",
    "read_image_to_tensor",
    "requires_module",
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


def contains_any_tensor(value: Any, dtype: Type = Tensor) -> bool:
    """
    Determine whether or not a list contains any Type
    """
    if isinstance(value, dtype):
        return True
    if isinstance(value, (list, tuple)):
        return any(contains_any_tensor(v, dtype=dtype) for v in value)
    elif isinstance(value, dict):
        return any(contains_any_tensor(v, dtype=dtype) for v in value.values())
    return False
