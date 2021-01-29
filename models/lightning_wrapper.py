# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
from torch import nn, Tensor

from torchvision.models.utils import load_state_dict_from_url

from . import yolo
from .transform import nested_tensor_from_tensor_list

from typing import Tuple, Any, List, Dict, Optional


class YOLOLitWrapper(nn.Module):
    """
    PyTorch Lightning implementation of `YOLO`
    """
    def __init__(
        self,
        arch: str = 'yolov5_darknet_pan_s_r31',
        learning_rate: float = 0.001,
        pretrained: bool = False,
        progress: bool = True,
        num_classes: int = 80,
        **kwargs: Any,
    ):
        """
        Args:
            arch: architecture
            learning_rate: the learning rate
            pretrained: if true, returns a model pre-trained on COCO train2017
            num_classes: number of detection classes (including background)
        """
        super().__init__()

        self.model = yolo.__dict__[arch](
            pretrained=pretrained, progress=progress, num_classes=num_classes, **kwargs)

    def forward(self, inputs: List[Tensor], targets: Optional[Tensor] = None,):
        sample = nested_tensor_from_tensor_list(inputs)
        return self.model(sample.tensors, targets=targets)
