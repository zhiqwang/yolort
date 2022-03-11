# Copyright (c) 2020, yolort team. All rights reserved.

import math
from typing import Tuple, Optional

import torch
from torch import nn, Tensor


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def encode_single(reference_boxes: Tensor, anchors: Tensor) -> Tensor:
    """
    Encode a set of anchors with respect to some reference boxes

    Args:
        reference_boxes (Tensor): reference boxes
        anchors_tuple (Tensor): boxes to be encoded
    """
    reference_boxes = torch.sigmoid(reference_boxes)

    pred_xy = reference_boxes[:, :2] * 2.0 - 0.5
    pred_wh = (reference_boxes[:, 2:4] * 2) ** 2 * anchors
    pred_boxes = torch.cat((pred_xy, pred_wh), 1)

    return pred_boxes


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


def bbox_iou(box1: Tensor, box2: Tensor, x1y1x2y2: bool = True, eps: float = 1e-7):
    """
    Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    """
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    # convex (smallest enclosing box) width
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    # convex height
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    # Complete IoU https://arxiv.org/abs/1911.08287v1
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    rho2 = (
        (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
    ) / 4  # center distance squared

    # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)  # CIoU


def smooth_binary_cross_entropy(eps: float = 0.1) -> Tuple[float, float]:
    # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing binary cross entropy targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(),
    # i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        # must be nn.BCEWithLogitsLoss()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        # required to apply FL to each element
        self.loss_fcn.reduction = "none"

    def forward(self, pred, logit):
        loss = self.loss_fcn(pred, logit)
        # p_t = torch.exp(-loss)
        # non-zero power for gradient stability
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma

        # TF implementation
        # https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = logit * pred_prob + (1 - logit) * (1 - pred_prob)
        alpha_factor = logit * self.alpha + (1 - logit) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        # 'none'
        return loss
