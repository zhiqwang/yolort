# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import unittest

from torch import nn

from yolort.utils import update_module_state_from_ultralytics


class UtilsTester(unittest.TestCase):
    def test_update_module_state_from_ultralytics(self):
        model = update_module_state_from_ultralytics(
            arch='yolov5s',
            version='r4.0',
            feature_fusion_type='PAN',
            num_classes=80,
            custom_path_or_model=None,
        )
        self.assertIsInstance(model, nn.Module)
