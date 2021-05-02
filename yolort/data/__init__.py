# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from .data_module import DetectionDataModule, VOCDataModule, COCODataModule
from .coco_eval import COCOEvaluator
from ._helper import contains_any_tensor
