# Copyright (c) 2022, yolort team. All rights reserved.

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from yolort.v5 import letterbox


class YOLOTransform:
    def __init__(
        self,
        height: int,
        width: int,
        *,
        size_divisible: int = 32,
        fixed_shape: Optional[Tuple[int, int]] = None,
        fill_color: Tuple[int, int, int] = (114, 114, 114),
        device: torch.device = torch.device("cpu"),
    ) -> None:

        self.height = height
        self.width = width
        self.size_divisible = size_divisible
        self.fixed_shape = fixed_shape
        self.fill_color = fill_color
        self.device = device

    def __call__(self, images):
        if isinstance(images, str):
            images = [images]
        images_info = [self.read_one_img(image) for image in images]
        images = [info[0].transpose([2, 0, 1]) for info in images_info]
        ratios = [info[1] for info in images_info]
        whs = [info[2] for info in images_info]
        return self.batch_images(images), ratios, whs

    def batch_images(self, images: List[np.ndarray]) -> Tensor:
        images = np.stack(images, 0)
        images = np.ascontiguousarray(images)
        images = images.astype(np.float32)
        images /= 255.0
        return torch.from_numpy(images).to(self.device)

    def read_one_img(self, image: str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, ratio, dwh = self.resize(image)
        return image, ratio, dwh

    def resize(self, image: np.ndarray):
        new_shape = (self.height, self.width)
        color = self.fill_color
        auto = not self.fixed_shape
        size_divisible = self.size_divisible
        return letterbox(image, new_shape, color=color, auto=auto, stride=size_divisible)
