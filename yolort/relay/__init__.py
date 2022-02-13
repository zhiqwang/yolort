# Copyright (c) 2021, yolort team. All rights reserved.

from .trace_wrapper import get_trace_module
from .trt_inference import YOLOTRTInference

__all__ = ["get_trace_module", "YOLOTRTInference"]
