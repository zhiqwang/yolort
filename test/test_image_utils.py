# Copyright (c) 2021, yolort team. All rights reserved.
import numpy as np
import torch

from yolort.utils.image_utils import (
    box_cxcywh_to_xyxy,
    cast_image_tensor_to_numpy,
    color_list,
    parse_images,
    parse_single_image,
    to_numpy,
)


class TestColorList:
    def test_returns_list(self):
        colors = color_list()
        assert isinstance(colors, list)
        assert len(colors) > 0

    def test_returns_rgb_tuples(self):
        colors = color_list()
        for c in colors:
            assert isinstance(c, tuple)
            assert len(c) == 3
            assert all(isinstance(v, int) for v in c)
            assert all(0 <= v <= 255 for v in c)


class TestToNumpy:
    def test_regular_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = to_numpy(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_tensor_with_grad(self):
        t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = to_numpy(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_2d_tensor(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = to_numpy(t)
        assert result.shape == (2, 2)


class TestBoxCxcywhToXyxy:
    def test_basic_conversion(self):
        bbox = np.array([[50.0, 50.0, 20.0, 30.0]])
        result = box_cxcywh_to_xyxy(bbox)
        expected = np.array([[40.0, 35.0, 60.0, 65.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_multiple_boxes(self):
        bbox = np.array([[50.0, 50.0, 20.0, 30.0], [100.0, 100.0, 40.0, 60.0]])
        result = box_cxcywh_to_xyxy(bbox)
        assert result.shape == (2, 4)
        # First box
        np.testing.assert_array_almost_equal(result[0], [40.0, 35.0, 60.0, 65.0])
        # Second box
        np.testing.assert_array_almost_equal(result[1], [80.0, 70.0, 120.0, 130.0])

    def test_zero_size_box(self):
        bbox = np.array([[50.0, 50.0, 0.0, 0.0]])
        result = box_cxcywh_to_xyxy(bbox)
        expected = np.array([[50.0, 50.0, 50.0, 50.0]])
        np.testing.assert_array_almost_equal(result, expected)


class TestCastImageTensorToNumpy:
    def test_basic_conversion(self):
        images = torch.rand(2, 3, 4, 4)
        result = cast_image_tensor_to_numpy(images)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.max() <= 255
        assert result.min() >= 0

    def test_output_range(self):
        # All ones -> 255
        images = torch.ones(1, 3, 4, 4)
        result = cast_image_tensor_to_numpy(images)
        assert np.all(result == 255)

    def test_output_zero(self):
        images = torch.zeros(1, 3, 4, 4)
        result = cast_image_tensor_to_numpy(images)
        assert np.all(result == 0)


class TestParseImages:
    def test_output_shape(self):
        images = torch.rand(2, 3, 32, 32)
        result = parse_images(images)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 32, 32, 3)  # NCHW -> NHWC
        assert result.dtype == np.uint8


class TestParseSingleImage:
    def test_output_shape(self):
        image = torch.rand(3, 32, 32)
        result = parse_single_image(image)
        assert isinstance(result, np.ndarray)
        assert result.shape == (32, 32, 3)  # CHW -> HWC
        assert result.dtype == np.uint8
