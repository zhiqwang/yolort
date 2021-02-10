import unittest
import numpy as np

import torch
from torch import Tensor

from yolort.utils.image_utils import box_cxcywh_to_xyxy, letterbox, read_image, scale_coords


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

    def test_box_cxcywh_to_xyxy(self):
        box_cxcywh = np.asarray([[50, 50, 100, 100],
                                 [0, 0, 0, 0],
                                 [20, 25, 20, 20],
                                 [58, 65, 70, 60]], dtype=np.float32)
        exp_xyxy = np.asarray([[0, 0, 100, 100],
                               [0, 0, 0, 0],
                               [10, 15, 30, 35],
                               [23, 35, 93, 95]], dtype=np.float32)

        box_xyxy = box_cxcywh_to_xyxy(box_cxcywh)
        self.assertEqual(exp_xyxy.shape, (4, 4))
        self.assertEqual(exp_xyxy.dtype, box_xyxy.dtype)
        self.assertIsNone(np.testing.assert_array_equal(exp_xyxy, box_xyxy))

    def test_scale_coords(self):
        box_tensor = torch.tensor([[0., 0., 100., 100.],
                                   [0., 0., 0., 0.],
                                   [10., 15., 30., 35.],
                                   [20., 35., 90., 95.]], dtype=torch.float)
        exp_coords = torch.tensor([[0., 0., 108.05, 111.25],
                                   [0., 0., 0., 0.],
                                   [7.9250, 16.6875, 30.1750, 38.9375],
                                   [19.05, 38.9375, 96.9250, 105.6875]], dtype=torch.float)
        TOLERANCE = 1e-5

        box_coords_scaled = scale_coords(box_tensor, (160, 128), (178, 136))
        self.assertEqual(tuple(box_coords_scaled.shape), (4, 4))
        self.assertTrue((exp_coords - box_coords_scaled).abs().max() < TOLERANCE)

if __name__ == '__main__':
    unittest.main()
