# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import unittest
import torch
from torch import nn

from yolort.utils import update_module_state_from_ultralytics


class UtilsTester(unittest.TestCase):
    @unittest.skipIf(float(torch.__version__[:3]) > 1.7, "ultralytics release 3.1 did't support torch 1.8+")
    def test_update_module_state_from_ultralytics_yolov5s_r31(self):
        model = update_module_state_from_ultralytics(arch='yolov5s', version='v3.1')
        self.assertIsInstance(model, nn.Module)

    @unittest.skipIf(float(torch.__version__[:3]) < 1.8, "Avoid multiple loading of the same model")
    def test_update_module_state_from_ultralytics_yolov5s_r40(self):
        model = update_module_state_from_ultralytics(arch='yolov5s', version='v4.0')
        self.assertIsInstance(model, nn.Module)
