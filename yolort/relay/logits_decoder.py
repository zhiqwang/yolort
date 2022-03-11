# Copyright (c) 2021, yolort team. All rights reserved.

from typing import List, Tuple

import torch
from torch import nn, Tensor
from yolort.models.box_head import _concat_pred_logits, _decode_pred_logits


class LogitsDecoder(nn.Module):
    """
    This is a simplified version of post-processing module, we manually remove
    the ``torchvision::ops::nms``, and it will be used later in the procedure for
    exporting the ONNX Graph to YOLOTRTModule or others.
    """

    def __init__(self, strides: List[int]) -> None:
        """
        Args:
            strides (List[int]): Strides of the AnchorGenerator.
        """

        super().__init__()
        self.strides = strides

    def forward(
        self,
        head_outputs: List[Tensor],
        grids: List[Tensor],
        shifts: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Just concat the predict logits, ignore the original ``torchvision::nms`` module
        from original ``yolort.models.box_head.PostProcess``.

        Args:
            head_outputs (List[Tensor]): The predicted locations and class/object confidence,
                shape of the element is (N, A, H, W, K).
            grids (List[Tensor]): Anchor grids.
            shifts (List[Tensor]): Anchor shifts.
        """
        batch_size = head_outputs[0].shape[0]
        device = head_outputs[0].device
        dtype = head_outputs[0].dtype
        strides = torch.as_tensor(self.strides, dtype=torch.float32, device=device).to(dtype=dtype)

        all_pred_logits = _concat_pred_logits(head_outputs, grids, shifts, strides)

        bbox_regression = []
        pred_scores = []

        for idx in range(batch_size):  # image idx, image inference
            pred_logits = all_pred_logits[idx]
            boxes, scores = _decode_pred_logits(pred_logits)
            bbox_regression.append(boxes)
            pred_scores.append(scores)

        # The default boxes tensor has shape [batch_size, number_boxes, 4].
        boxes = torch.stack(bbox_regression)
        scores = torch.stack(pred_scores)
        return boxes, scores
