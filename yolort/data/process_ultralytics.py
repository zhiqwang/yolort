# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..utils.image_utils import letterbox, scale_coords

from typing import Dict, Optional, List, Tuple, Union


class AutoShape(nn.Module):
    """
    Make letter box transform to images and bounding box targets.
    """
    def __init__(self, size: Union[int, Tuple[int]]):
        """
        Args:
            size (int or tuple of int): the size of the transformed images.
        """
        super().__init__()
        self.size = size
        if isinstance(size, int):
            self.size = (size, size)

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]],
    ):
        """
        Args:
            images (List[Tensor]): images to be processed.
            targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
        """
        images = letterbox(images)
        return images, targets

    def postprocess(
        self,
        predictions: Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]],
        image_shapes: List[Tuple[int, int]],
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, Tensor]]:

        for i, (pred, im_s, o_im_s) in enumerate(zip(predictions, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = scale_coords(boxes, im_s, o_im_s)
            predictions[i]["boxes"] = boxes

        return predictions
