# Modified from ultralytics/yolov5 by Zhiqiang Wang
import math

import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.ops import box_convert

from typing import Tuple


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(
        self,
        weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        bbox_xform_clip: float = math.log(1000. / 16),
    ) -> None:
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def decode_single(
        self,
        rel_codes: Tensor,
        anchors_tuple: Tuple[Tensor, Tensor, Tensor],
    ):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            anchors_tupe (Tensor, Tensor, Tensor): reference boxes.
        """

        pred_wh = (rel_codes[..., 0:2] * 2. + anchors_tuple[0]) * anchors_tuple[1]  # wh
        pred_xy = (rel_codes[..., 2:4] * 2) ** 2 * anchors_tuple[2]  # xy
        pred_boxes = torch.cat([pred_wh, pred_xy], dim=1)
        pred_boxes = box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")

        return pred_boxes


def bbox_ciou(box1, box2, eps: float = 1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    # transform from xywh to xyxy
    b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
    b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    # center distance squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4

    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / ((1 + eps) - iou + v)

    return iou - (rho2 / c2 + v * alpha)  # CIoU


def cls_loss(inputs, targets, pos_weight):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight)
    return loss


def obj_loss(inputs, targets, pos_weight):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight)
    return loss
