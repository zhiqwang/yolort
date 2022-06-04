# Copyright (c) 2022, yolort team. All rights reserved.

import random

import torch
from torch import Tensor


class NonMaxSupressionOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        boxes: Tensor,
        scores: Tensor,
        max_output_boxes_per_class: Tensor = torch.tensor([100]),
        iou_threshold: Tensor = torch.tensor([0.45]),
        score_threshold: Tensor = torch.tensor([0.35]),
    ):
        """
        Args:
            boxes (Tensor): An input tensor with shape [num_batches, spatial_dimension, 4].
                have been multiplied max_wh here.
            scores (Tensor): An input tensor with shape [num_batches, num_classes, spatial_dimension].
                only one class score here.
            max_output_boxes_per_class (Tensor, optional): Integer representing the maximum number of
                boxes to be selected per batch per class. It is a scalar.
            iou_threshold (Tensor, optional): Float representing the threshold for deciding whether
                boxes overlap too much with respect to IOU. It is scalar. Value range [0, 1].
            score_threshold (Tensor, optional): Float representing the threshold for deciding when to
                remove boxes based on score. It is a scalar.
        """
        batch = scores.shape[0]
        num_det = random.randint(0, 100)
        batches = torch.randint(0, batch, (num_det,)).sort()[0]
        idxs = torch.arange(100, 100 + num_det)
        zeros = torch.zeros((num_det,), dtype=torch.int64)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)
        return selected_indices

    @staticmethod
    def symbolic(
        g,
        boxes: Tensor,
        scores: Tensor,
        max_output_boxes_per_class: Tensor,
        iou_threshold: Tensor,
        score_threshold: Tensor,
    ):
        return g.op(
            "NonMaxSuppression",
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
        )


class EfficientNMSOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        boxes: Tensor,
        scores: Tensor,
        background_class: int = -1,
        box_coding: int = 0,
        iou_threshold: float = 0.45,
        max_output_boxes: int = 100,
        plugin_version: str = "1",
        score_activation: int = 0,
        score_threshold: float = 0.35,
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1))
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes))

        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(
        g,
        boxes: Tensor,
        scores: Tensor,
        background_class: int = -1,
        box_coding: int = 0,
        iou_threshold: float = 0.45,
        max_output_boxes: int = 100,
        plugin_version: str = "1",
        score_activation: int = 0,
        score_threshold: float = 0.35,
    ):

        return g.op(
            "TRT::EfficientNMS_TRT",
            boxes,
            scores,
            background_class_i=background_class,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            plugin_version_s=plugin_version,
            score_activation_i=score_activation,
            score_threshold_f=score_threshold,
            outputs=4,
        )
