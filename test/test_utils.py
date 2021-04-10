# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import unittest
import numpy as np

from torch import nn, Tensor

from yolort.utils import (
    update_module_state_from_ultralytics,
    read_image_to_tensor,
    get_image_from_url,
)


class UtilsTester(unittest.TestCase):
    def test_update_module_state_from_ultralytics(self):
        model = update_module_state_from_ultralytics(
            arch='yolov5s',
            version='v4.0',
            feature_fusion_type='PAN',
            num_classes=80,
            custom_path_or_model=None,
        )
        self.assertIsInstance(model, nn.Module)

    def test_read_image_to_tensor(self):
        N, H, W = 3, 720, 360
        img = np.random.randint(0, 255, (H, W, N), dtype='uint8')  # As a dummy image
        out = read_image_to_tensor(img)

        self.assertIsInstance(out, Tensor)
        self.assertEqual(tuple(out.shape), (N, H, W))

    def test_get_image_from_url(self):
        url = "https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg"
        img = get_image_from_url(url)
        self.assertIsInstance(img, np.ndarray)
        self.assertTupleEqual(img.shape, (720, 1280, 3))
