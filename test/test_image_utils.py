# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import numpy as np

import torch

from yolort.utils.image_utils import box_cxcywh_to_xyxy
from yolort.v5 import letterbox, scale_coords


def test_letterbox():
    img = np.random.randint(0, 255, (720, 360, 3), dtype='uint8')  # As a dummy image
    out = letterbox(img, new_shape=(416, 416))[0]
    assert tuple(out.shape) == (416, 224, 3)


def test_box_cxcywh_to_xyxy():
    box_cxcywh = np.asarray([[50, 50, 100, 100],
                            [0, 0, 0, 0],
                            [20, 25, 20, 20],
                            [58, 65, 70, 60]], dtype=np.float)
    exp_xyxy = np.asarray([[0, 0, 100, 100],
                           [0, 0, 0, 0],
                           [10, 15, 30, 35],
                           [23, 35, 93, 95]], dtype=np.float)

    box_xyxy = box_cxcywh_to_xyxy(box_cxcywh)
    assert exp_xyxy.shape == (4, 4)
    assert exp_xyxy.dtype == box_xyxy.dtype
    np.testing.assert_array_equal(exp_xyxy, box_xyxy)


def test_scale_coords():
    box_tensor = torch.tensor([[0., 0., 100., 100.],
                               [0., 0., 0., 0.],
                               [10., 15., 30., 35.],
                               [20., 35., 90., 95.]], dtype=torch.float)
    exp_coords = torch.tensor([[0., 0., 108.05, 111.25],
                               [0., 0., 0., 0.],
                               [7.9250, 16.6875, 30.1750, 38.9375],
                               [19.05, 38.9375, 96.9250, 105.6875]], dtype=torch.float)

    box_coords_scaled = scale_coords((160, 128), box_tensor, (178, 136))
    assert tuple(box_coords_scaled.shape) == (4, 4)
    torch.testing.assert_close(box_coords_scaled, exp_coords)
