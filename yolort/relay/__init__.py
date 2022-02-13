# Copyright (c) 2021, yolort team. All rights reserved.

from .trace_wrapper import get_trace_module
from .trt_inference import YOLOInference

__all__ = ["get_trace_module", "YOLOInference"]
