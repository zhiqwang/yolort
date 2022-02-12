# Copyright (c) 2021, yolort team. All rights reserved.

from .y_onnxruntime import PredictorORT
from .y_tensorrt import PredictorTRT

__all__ = ["PredictorORT", "PredictorTRT"]
