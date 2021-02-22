# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import unittest
import torch
from torch import nn

from yolort.utils import update_module_state_from_ultralytics


class UtilsTester(unittest.TestCase):
    def test_update_module_state_from_ultralytics(self):
        model = update_module_state_from_ultralytics(arch='yolov5s', version='v4.0')
        self.assertIsInstance(model, nn.Module)
