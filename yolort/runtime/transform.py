from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor


class YOLOTransform:
    def __init__(
        self,
        size_divisible: int = 32,
        fixed_shape: Optional[Tuple[int, int]] = None,
        fill_color: int = 114,
        device: torch.device = torch.device("cpu"),
    ) -> None:

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
        image, ratio, dwh = self.resize(image, self.fixed_shape)
        return image, ratio, dwh

    def resize(self, image: np.ndarray, new_shape: Tuple[int, int] = (320, 320)):
        shape = image.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(np.round(shape[1] * r)), int(np.round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(np.round(dh - 0.1)), int(np.round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(self.fill_color,) * 3
        )
        return image, ratio, (dw, dh)
