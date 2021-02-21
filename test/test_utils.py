import unittest
from torch import nn

from yolort.utils import update_module_state_from_ultralytics


class UtilsTester(unittest.TestCase):
    def test_update_module_state_from_ultralytics_yolov5s(self):
        model = update_module_state_from_ultralytics(arch='yolov5s', release='v3.1')
        self.assertIsInstance(model, nn.Module)


if __name__ == '__main__':
    unittest.main()
