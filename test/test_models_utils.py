# Copyright (c) 2021, yolort team. All rights reserved.
import pytest
import torch
from torch import nn

from yolort.models._utils import (
    FocalLoss,
    _make_divisible,
    bbox_iou,
    decode_single,
    encode_single,
    smooth_binary_cross_entropy,
)


class TestMakeDivisible:
    def test_basic(self):
        assert _make_divisible(16.0, 8) == 16

    def test_round_up(self):
        assert _make_divisible(17.0, 8) == 16  # 17 + 4 = 21 // 8 * 8 = 16

    def test_min_value(self):
        assert _make_divisible(1.0, 8, min_value=8) == 8

    def test_custom_min_value(self):
        assert _make_divisible(1.0, 8, min_value=16) == 16

    def test_10_percent_rule(self):
        # If new_v < 0.9 * v, add divisor
        result = _make_divisible(20.0, 16)
        assert result >= 0.9 * 20.0

    def test_large_value(self):
        result = _make_divisible(256.0, 8)
        assert result == 256
        assert result % 8 == 0


class TestEncodeSingle:
    def test_basic_encode(self):
        ref_boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        anchors = torch.tensor([[10.0, 10.0]])
        result = encode_single(ref_boxes, anchors)
        assert result.shape == (1, 4)

    def test_output_shape(self):
        ref_boxes = torch.randn(5, 4)
        anchors = torch.randn(5, 2)
        result = encode_single(ref_boxes, anchors)
        assert result.shape == (5, 4)


class TestDecodeSingle:
    def test_basic_decode(self):
        rel_codes = torch.randn(1, 5, 5, 4)
        grid = torch.zeros(1, 5, 5, 2)
        shift = torch.ones(1, 5, 5, 2) * 10.0
        stride = torch.tensor([8.0])
        pred_xy, pred_wh = decode_single(rel_codes, grid, shift, stride)
        assert pred_xy.shape == (1, 5, 5, 2)
        assert pred_wh.shape == (1, 5, 5, 2)


class TestBboxIou:
    def test_identical_boxes(self):
        box1 = torch.tensor([0.0, 0.0, 10.0, 10.0])
        box2 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        result = bbox_iou(box1, box2, x1y1x2y2=True)
        # CIoU for identical boxes should be close to 1
        assert result.item() == pytest.approx(1.0, abs=1e-3)

    def test_xywh_format(self):
        box1 = torch.tensor([5.0, 5.0, 10.0, 10.0])  # center x, y, w, h
        box2 = torch.tensor([[5.0, 5.0, 10.0, 10.0]])
        result = bbox_iou(box1, box2, x1y1x2y2=False)
        assert result.item() == pytest.approx(1.0, abs=1e-3)

    def test_no_overlap(self):
        box1 = torch.tensor([0.0, 0.0, 10.0, 10.0])
        box2 = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
        result = bbox_iou(box1, box2, x1y1x2y2=True)
        # Should be negative due to CIoU penalty for non-overlapping boxes
        assert result.item() < 0


class TestSmoothBinaryCrossEntropy:
    def test_default_eps(self):
        pos, neg = smooth_binary_cross_entropy(eps=0.1)
        assert pos == pytest.approx(0.95, abs=1e-6)
        assert neg == pytest.approx(0.05, abs=1e-6)

    def test_zero_eps(self):
        pos, neg = smooth_binary_cross_entropy(eps=0.0)
        assert pos == pytest.approx(1.0, abs=1e-6)
        assert neg == pytest.approx(0.0, abs=1e-6)

    def test_full_smoothing(self):
        pos, neg = smooth_binary_cross_entropy(eps=1.0)
        assert pos == pytest.approx(0.5, abs=1e-6)
        assert neg == pytest.approx(0.5, abs=1e-6)


class TestFocalLoss:
    def test_output_shape_mean(self):
        loss_fcn = nn.BCEWithLogitsLoss(reduction="mean")
        focal = FocalLoss(loss_fcn, gamma=1.5, alpha=0.25)
        pred = torch.randn(10, 5)
        target = torch.zeros(10, 5)
        loss = focal(pred, target)
        assert loss.shape == ()  # scalar

    def test_output_shape_none(self):
        loss_fcn = nn.BCEWithLogitsLoss(reduction="none")
        focal = FocalLoss(loss_fcn, gamma=1.5, alpha=0.25)
        pred = torch.randn(10, 5)
        target = torch.zeros(10, 5)
        loss = focal(pred, target)
        assert loss.shape == (10, 5)

    def test_output_shape_sum(self):
        loss_fcn = nn.BCEWithLogitsLoss(reduction="sum")
        focal = FocalLoss(loss_fcn, gamma=1.5, alpha=0.25)
        pred = torch.randn(10, 5)
        target = torch.zeros(10, 5)
        loss = focal(pred, target)
        assert loss.shape == ()  # scalar

    def test_positive_loss(self):
        loss_fcn = nn.BCEWithLogitsLoss(reduction="mean")
        focal = FocalLoss(loss_fcn)
        pred = torch.randn(10, 5)
        target = torch.zeros(10, 5)
        loss = focal(pred, target)
        assert loss.item() >= 0
