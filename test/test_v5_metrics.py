# Copyright (c) 2021, yolort team. All rights reserved.
import numpy as np
import pytest
import torch

from yolort.v5.utils.metrics import bbox_ioa, bbox_iou, box_iou, compute_ap, fitness, wh_iou


class TestFitness:
    def test_basic_fitness(self):
        # x is (n, 4+) where columns are [P, R, mAP@0.5, mAP@0.5:0.95]
        x = np.array([[0.9, 0.8, 0.7, 0.6]])
        result = fitness(x)
        # weights = [0.0, 0.0, 0.1, 0.9]
        expected = 0.0 * 0.9 + 0.0 * 0.8 + 0.1 * 0.7 + 0.9 * 0.6
        assert result[0] == pytest.approx(expected, abs=1e-6)

    def test_fitness_multiple(self):
        x = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
        result = fitness(x)
        assert result[0] == pytest.approx(1.0, abs=1e-6)
        assert result[1] == pytest.approx(0.0, abs=1e-6)

    def test_fitness_dominated_by_map5095(self):
        x = np.array([[0.0, 0.0, 0.0, 1.0]])
        result = fitness(x)
        assert result[0] == pytest.approx(0.9, abs=1e-6)


class TestBoxIou:
    def test_identical_boxes(self):
        box1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        box2 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        result = box_iou(box1, box2)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_no_overlap(self):
        box1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        box2 = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
        result = box_iou(box1, box2)
        assert result[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_partial_overlap(self):
        box1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        box2 = torch.tensor([[5.0, 5.0, 15.0, 15.0]])
        result = box_iou(box1, box2)
        # Intersection = 5*5 = 25, Union = 100 + 100 - 25 = 175
        expected = 25.0 / 175.0
        assert result[0, 0] == pytest.approx(expected, abs=1e-4)

    def test_box_iou_multiple(self):
        box1 = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
        box2 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        result = box_iou(box1, box2)
        assert result.shape == (2, 1)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert result[1, 0] == pytest.approx(0.0, abs=1e-6)


class TestBboxIou:
    def test_identical_boxes_xyxy(self):
        box1 = torch.tensor([0.0, 0.0, 10.0, 10.0])
        box2 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        result = bbox_iou(box1, box2, x1y1x2y2=True)
        assert result.item() == pytest.approx(1.0, abs=1e-4)

    def test_identical_boxes_xywh(self):
        box1 = torch.tensor([5.0, 5.0, 10.0, 10.0])
        box2 = torch.tensor([[5.0, 5.0, 10.0, 10.0]])
        result = bbox_iou(box1, box2, x1y1x2y2=False)
        assert result.item() == pytest.approx(1.0, abs=1e-4)

    def test_no_overlap_xyxy(self):
        box1 = torch.tensor([0.0, 0.0, 10.0, 10.0])
        box2 = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
        # Without GIoU/DIoU/CIoU flags, returns plain IoU
        result = bbox_iou(box1, box2, x1y1x2y2=True)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_giou_identical(self):
        box1 = torch.tensor([0.0, 0.0, 10.0, 10.0])
        box2 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        result = bbox_iou(box1, box2, x1y1x2y2=True, GIoU=True)
        assert result.item() == pytest.approx(1.0, abs=1e-4)

    def test_giou_no_overlap(self):
        box1 = torch.tensor([0.0, 0.0, 10.0, 10.0])
        box2 = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
        result = bbox_iou(box1, box2, x1y1x2y2=True, GIoU=True)
        # GIoU is less than IoU for non-overlapping boxes
        assert result.item() < 0

    def test_diou(self):
        box1 = torch.tensor([0.0, 0.0, 10.0, 10.0])
        box2 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        result = bbox_iou(box1, box2, x1y1x2y2=True, DIoU=True)
        assert result.item() == pytest.approx(1.0, abs=1e-4)

    def test_ciou(self):
        box1 = torch.tensor([0.0, 0.0, 10.0, 10.0])
        box2 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        result = bbox_iou(box1, box2, x1y1x2y2=True, CIoU=True)
        assert result.item() == pytest.approx(1.0, abs=1e-4)


class TestBboxIoa:
    def test_identical_boxes(self):
        box1 = np.array([0.0, 0.0, 10.0, 10.0])
        box2 = np.array([[0.0, 0.0, 10.0, 10.0]])
        result = bbox_ioa(box1, box2)
        assert result[0] == pytest.approx(1.0, abs=1e-6)

    def test_box1_contains_box2(self):
        box1 = np.array([0.0, 0.0, 20.0, 20.0])
        box2 = np.array([[5.0, 5.0, 10.0, 10.0]])
        result = bbox_ioa(box1, box2)
        # Intersection = 5*5 = 25, box2 area = 5*5 = 25
        assert result[0] == pytest.approx(1.0, abs=1e-6)

    def test_no_overlap(self):
        box1 = np.array([0.0, 0.0, 5.0, 5.0])
        box2 = np.array([[10.0, 10.0, 20.0, 20.0]])
        result = bbox_ioa(box1, box2)
        assert result[0] == pytest.approx(0.0, abs=1e-6)


class TestWhIou:
    def test_identical_wh(self):
        wh1 = torch.tensor([[10.0, 10.0]])
        wh2 = torch.tensor([[10.0, 10.0]])
        result = wh_iou(wh1, wh2)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_different_wh(self):
        wh1 = torch.tensor([[10.0, 10.0]])
        wh2 = torch.tensor([[20.0, 20.0]])
        result = wh_iou(wh1, wh2)
        # min_w = 10, min_h = 10, inter = 100
        # area1 = 100, area2 = 400, union = 400
        expected = 100.0 / 400.0
        assert result[0, 0] == pytest.approx(expected, abs=1e-4)


class TestComputeAp:
    @pytest.mark.skipif(
        not hasattr(np, "trapz"),
        reason="np.trapz removed in NumPy 2.0 (pre-existing compat issue)",
    )
    def test_perfect_precision_recall(self):
        recall = np.array([0.0, 0.5, 1.0])
        precision = np.array([1.0, 1.0, 1.0])
        ap, _, _ = compute_ap(recall, precision)
        assert ap == pytest.approx(1.0, abs=1e-2)

    @pytest.mark.skipif(
        not hasattr(np, "trapz"),
        reason="np.trapz removed in NumPy 2.0 (pre-existing compat issue)",
    )
    def test_zero_precision(self):
        recall = np.array([0.0, 0.5, 1.0])
        precision = np.array([0.0, 0.0, 0.0])
        ap, _, _ = compute_ap(recall, precision)
        assert ap == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.skipif(
        not hasattr(np, "trapz"),
        reason="np.trapz removed in NumPy 2.0 (pre-existing compat issue)",
    )
    def test_returns_three_values(self):
        recall = np.array([0.0, 0.5, 1.0])
        precision = np.array([1.0, 0.5, 0.0])
        result = compute_ap(recall, precision)
        assert len(result) == 3  # ap, mpre, mrec
