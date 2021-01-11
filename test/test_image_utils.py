import unittest
import numpy as np
from torch import Tensor

from utils.image_utils import letterbox, read_image


class ImageUtilsTester(unittest.TestCase):
    def test_read_image(self):
        N, H, W = 3, 720, 360
        img = np.random.randint(0, 255, (H, W, N), dtype='uint8')  # As a dummy image
        out = read_image(img)

        self.assertIsInstance(out, Tensor)
        self.assertEqual(tuple(out.shape), (N, H, W))


    def test_letterbox(self):
        img = np.random.randint(0, 255, (720, 360, 3), dtype='uint8')  # As a dummy image
        out = letterbox(img, new_shape=(416, 416))[0]
        self.assertEqual(tuple(out.shape), (416, 224, 3))


if __name__ == '__main__':
    unittest.main()
