# Copyright (c) 2021, yolort team. All rights reserved.

from typing import List, Tuple

import torch
from torch import nn, Tensor
from torchvision.ops import box_convert

def _decode_pred_logits(pred_logits: Tensor):
    """
    Decode the prediction logit from the PostPrecess.
    """
    # Compute conf
    # box_conf x class_conf, w/ shape: num_anchors x num_classes
    scores = pred_logits[..., 5:] * pred_logits[..., 4:5]
    boxes = box_convert(pred_logits[..., :4], in_fmt="cxcywh", out_fmt="xyxy")

    return boxes, scores

def decode_single(
    rel_codes: Tensor,
    grid: Tensor,
    shift: Tensor,
    stride: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    From a set of original boxes and encoded relative box offsets,
    get the decoded boxes.

    Args:
        rel_codes (Tensor): Encoded boxes
        grid (Tensor): Anchor grids
        shift (Tensor): Anchor shifts
        stride (int): Stride
    """
    pred_xy = (rel_codes[..., 0:2] * 2.0 - 0.5 + grid) * stride
    pred_wh = (rel_codes[..., 2:4] * 2.0) ** 2 * shift

    return pred_xy, pred_wh

def _concat_pred_logits(
    head_outputs: List[Tensor],
    grids: List[Tensor],
    shifts: List[Tensor],
    strides: Tensor,
) -> Tensor:
    # Concat all pred logits
    batch_size, _, _, _, K = head_outputs[0].shape

    # Decode bounding box with the shifts and grids
    all_pred_logits = []

    for head_output, grid, shift, stride in zip(head_outputs, grids, shifts, strides):
        head_feature = torch.sigmoid(head_output)
        pred_xy, pred_wh = det_utils.decode_single(head_feature[..., :4], grid, shift, stride)
        pred_logits = torch.cat((pred_xy, pred_wh, head_feature[..., 4:]), dim=-1)
        all_pred_logits.append(pred_logits.view(batch_size, -1, K))

    all_pred_logits = torch.cat(all_pred_logits, dim=1)

    return all_pred_logits

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
