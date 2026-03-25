# Copyright (c) 2021, yolort team. All rights reserved.
import numpy as np
import pytest
import torch

from yolort.v5.utils.general import (
    check_img_size,
    check_suffix,
    clean_str,
    clip_coords,
    colorstr,
    increment_path,
    intersect_dicts,
    is_ascii,
    is_chinese,
    make_divisible,
    one_cycle,
    url2file,
    xywh2xyxy,
    xywhn2xyxy,
    xyxy2xywh,
    xyxy2xywhn,
)


class TestCoordinateConversions:
    """Tests for box coordinate conversion utilities."""

    def test_xyxy2xywh_tensor(self):
        boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        result = xyxy2xywh(boxes)
        expected = torch.tensor([[20.0, 30.0, 20.0, 20.0]])
        torch.testing.assert_close(result, expected)

    def test_xyxy2xywh_numpy(self):
        boxes = np.array([[10.0, 20.0, 30.0, 40.0]])
        result = xyxy2xywh(boxes)
        expected = np.array([[20.0, 30.0, 20.0, 20.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_xywh2xyxy_tensor(self):
        boxes = torch.tensor([[20.0, 30.0, 20.0, 20.0]])
        result = xywh2xyxy(boxes)
        expected = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        torch.testing.assert_close(result, expected)

    def test_xywh2xyxy_numpy(self):
        boxes = np.array([[20.0, 30.0, 20.0, 20.0]])
        result = xywh2xyxy(boxes)
        expected = np.array([[10.0, 20.0, 30.0, 40.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_xyxy2xywh_roundtrip(self):
        boxes = torch.tensor([[10.0, 20.0, 50.0, 60.0], [0.0, 0.0, 100.0, 200.0]])
        result = xywh2xyxy(xyxy2xywh(boxes))
        torch.testing.assert_close(result, boxes)

    def test_xywhn2xyxy_default_size(self):
        # Normalized box [x_center=0.5, y_center=0.5, w=0.5, h=0.5] in 640x640 image
        boxes = np.array([[0.5, 0.5, 0.5, 0.5]])
        result = xywhn2xyxy(boxes, w=640, h=640)
        expected = np.array([[160.0, 160.0, 480.0, 480.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_xywhn2xyxy_with_padding(self):
        boxes = np.array([[0.5, 0.5, 1.0, 1.0]])
        result = xywhn2xyxy(boxes, w=100, h=100, padw=10, padh=20)
        expected = np.array([[10.0, 20.0, 110.0, 120.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_xyxy2xywhn_default(self):
        boxes = np.array([[160.0, 160.0, 480.0, 480.0]])
        result = xyxy2xywhn(boxes, w=640, h=640)
        expected = np.array([[0.5, 0.5, 0.5, 0.5]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_xyxy2xywh_multiple_boxes(self):
        boxes = torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [5.0, 5.0, 15.0, 25.0],
            ]
        )
        result = xyxy2xywh(boxes)
        expected = torch.tensor(
            [
                [5.0, 5.0, 10.0, 10.0],
                [10.0, 15.0, 10.0, 20.0],
            ]
        )
        torch.testing.assert_close(result, expected)


class TestClipCoords:
    def test_clip_coords_tensor(self):
        boxes = torch.tensor([[-5.0, -10.0, 650.0, 500.0]])
        clip_coords(boxes, (480, 640))
        expected = torch.tensor([[0.0, 0.0, 640.0, 480.0]])
        torch.testing.assert_close(boxes, expected)

    def test_clip_coords_numpy(self):
        boxes = np.array([[-5.0, -10.0, 650.0, 500.0]])
        clip_coords(boxes, (480, 640))
        expected = np.array([[0.0, 0.0, 640.0, 480.0]])
        np.testing.assert_array_almost_equal(boxes, expected)

    def test_clip_coords_within_bounds(self):
        boxes = torch.tensor([[10.0, 20.0, 100.0, 200.0]])
        clip_coords(boxes, (480, 640))
        expected = torch.tensor([[10.0, 20.0, 100.0, 200.0]])
        torch.testing.assert_close(boxes, expected)


class TestMakeDivisible:
    def test_already_divisible(self):
        assert make_divisible(64, 32) == 64

    def test_not_divisible(self):
        assert make_divisible(65, 32) == 96

    def test_smaller_than_divisor(self):
        assert make_divisible(10, 32) == 32

    def test_zero(self):
        assert make_divisible(0, 32) == 0


class TestCheckImgSize:
    def test_already_multiple(self):
        assert check_img_size(640, stride=32) == 640

    def test_not_multiple_int(self):
        result = check_img_size(641, stride=32)
        assert result % 32 == 0
        assert result >= 641

    def test_list_input(self):
        result = check_img_size([641, 481], stride=32)
        assert isinstance(result, list)
        assert all(r % 32 == 0 for r in result)


class TestStringUtils:
    def test_is_ascii_true(self):
        assert is_ascii("hello world") is True

    def test_is_ascii_false(self):
        assert is_ascii("héllo") is False

    def test_is_ascii_empty(self):
        assert is_ascii("") is True

    def test_is_chinese_true(self):
        assert is_chinese("人工智能") is not None

    def test_is_chinese_false(self):
        assert is_chinese("hello") is None

    def test_clean_str(self):
        assert clean_str("hello@world#test") == "hello_world_test"

    def test_clean_str_no_special(self):
        assert clean_str("hello_world") == "hello_world"

    def test_colorstr_basic(self):
        result = colorstr("blue", "bold", "hello")
        assert "hello" in result
        assert "\033[" in result

    def test_colorstr_single_arg(self):
        result = colorstr("hello")
        assert "hello" in result

    def test_url2file(self):
        url = "https://example.com/path/to/file.txt?auth=token"
        result = url2file(url)
        assert result == "file.txt"


class TestCheckSuffix:
    def test_valid_suffix(self):
        # Should not raise
        check_suffix("model.pt", (".pt",))

    def test_invalid_suffix(self):
        with pytest.raises(AssertionError):
            check_suffix("model.onnx", (".pt",))

    def test_empty_file(self):
        # Should not raise for empty file
        check_suffix("", (".pt",))

    def test_list_of_files(self):
        # Should not raise for valid files
        check_suffix(["model1.pt", "model2.pt"], (".pt",))


class TestIntersectDicts:
    def test_basic_intersection(self):
        d1 = {"a": torch.tensor([1, 2]), "b": torch.tensor([3, 4])}
        d2 = {"a": torch.tensor([5, 6]), "c": torch.tensor([7, 8])}
        result = intersect_dicts(d1, d2)
        assert "a" in result
        assert "b" not in result
        assert "c" not in result

    def test_shape_mismatch(self):
        d1 = {"a": torch.tensor([1, 2, 3])}
        d2 = {"a": torch.tensor([1, 2])}
        result = intersect_dicts(d1, d2)
        assert len(result) == 0

    def test_exclude_keys(self):
        d1 = {"anchor": torch.tensor([1.0]), "weight": torch.tensor([2.0])}
        d2 = {"anchor": torch.tensor([3.0]), "weight": torch.tensor([4.0])}
        result = intersect_dicts(d1, d2, exclude=("anchor",))
        assert "anchor" not in result
        assert "weight" in result


class TestOneCycle:
    def test_one_cycle_start(self):
        fn = one_cycle(y1=0.0, y2=1.0, steps=100)
        assert fn(0) == pytest.approx(0.0, abs=1e-6)

    def test_one_cycle_end(self):
        # Sinusoidal ramp from y1 to y2: at x=steps, value is y2
        fn = one_cycle(y1=0.0, y2=1.0, steps=100)
        assert fn(100) == pytest.approx(1.0, abs=1e-6)

    def test_one_cycle_middle(self):
        # At half-way, the value is midpoint between y1 and y2
        fn = one_cycle(y1=0.0, y2=1.0, steps=100)
        assert fn(50) == pytest.approx(0.5, abs=1e-6)


class TestIncrementPath:
    def test_increment_new_path(self, tmp_path):
        result = increment_path(tmp_path / "exp")
        assert str(result).endswith("exp")

    def test_increment_existing_path(self, tmp_path):
        (tmp_path / "exp").mkdir()
        result = increment_path(tmp_path / "exp")
        assert str(result).endswith("exp2")

    def test_increment_exist_ok(self, tmp_path):
        (tmp_path / "exp").mkdir()
        result = increment_path(tmp_path / "exp", exist_ok=True)
        assert str(result).endswith("exp")
