# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import unittest
import torch
from torch import nn

from yolort.utils import update_module_state_from_ultralytics


class UtilsTester(unittest.TestCase):
    @unittest.skipIf(float(torch.__version__[:3]) > 1.7, "ultralytics release 3.1 did't support torch 1.8+")
    def test_update_module_state_from_ultralytics_yolov5s(self):
        model = update_module_state_from_ultralytics(arch='yolov5s', release='v3.1')
        self.assertIsInstance(model, nn.Module)


if __name__ == '__main__':
    unittest.main()
