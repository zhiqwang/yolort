# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from typing import Tuple, Union


class LetterBox(nn.Module):
    """
    Make letter box transform to image and bounding box target.

    https://github.com/pytorch/vision/issues/3286#issue-792821822

    Args:
        size (int or tuple of int): the size of the transformed image.
    """

    def __init__(self, size: Union[int, Tuple[int]]):
        super().__init__()
        self.size = size
        if isinstance(size, int):
            self.size = (size, size)

    def forward(self, image: Tensor, target: Union[np.ndarray, Tensor]):
        """
        Args:
            image (PIL Image): Image to be transformed.
            target (np.ndarray or Tensor): bounding box target to be transformed.

        Returns:
            tuple: (image, target)
        """
        old_width, old_height = image.size
        width, height = self.size

        ratio = min(width / old_width, height / old_height)
        new_width = int(old_width * ratio)
        new_height = int(old_height * ratio)
        image = F.resize(image, (new_height, new_width))

        pad_x = (width - new_width) * 0.5
        pad_y = (height - new_height) * 0.5
        left, right = round(pad_x + 0.1), round(pad_x - 0.1)
        top, bottom = round(pad_y + 0.1), round(pad_y - 0.1)
        padding = (left, top, right, bottom)
        image = F.pad(image, padding, 255 // 2)

        target[..., 0] = torch.round(ratio * target[..., 0]) + left
        target[..., 1] = torch.round(ratio * target[..., 1]) + top
        target[..., 2] = torch.round(ratio * target[..., 2]) + right
        target[..., 3] = torch.round(ratio * target[..., 3]) + bottom
        return image, target
