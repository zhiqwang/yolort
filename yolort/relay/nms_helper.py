# Copyright (c) 2022, yolort team. All rights reserved.

import random

import torch
import torchvision
from torch import Tensor


def batched_nms(
    prediction: Tensor,
    score_thresh: float = 0.25,
    nms_thresh: float = 0.45,
    agnostic: bool = False,
):
    """
    Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
        list of detections, on (n, 6) tensor per image [xyxy, conf, cls]
    """
    max_wh = 4096  # (pixels) maximum box width and height
    xc = prediction[..., 4] > score_thresh  # candidates
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        # Compute conf
        cx, cy, w, h = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        obj_conf = x[:, 4:5]
        cls_conf = x[:, 5:]
        cls_conf = obj_conf * cls_conf  # conf = obj_conf * cls_conf
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(cx, cy, w, h)
        conf, j = cls_conf.max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > score_thresh]
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, nms_thresh)  # NMS
        output[xi] = x[i]
    return output


def xywh2xyxy(cx, cy, w, h):
    """
    This function is used while exporting ONNX models

    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    """
    halfw = w / 2
    halfh = h / 2
    xmin = cx - halfw  # top left x
    ymin = cy - halfh  # top left y
    xmax = cx + halfw  # bottom right x
    ymax = cy + halfh  # bottom right y
    return torch.cat((xmin, ymin, xmax, ymax), 1)


class NonMaxSupressionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, boxes, scores, detections_per_class, iou_thresh, score_thresh):
        """
        Symbolic method to export an NonMaxSupression ONNX models.

        Args:
            boxes (Tensor): An input tensor with shape [num_batches, spatial_dimension, 4].
                have been multiplied original size here.
            scores (Tensor): An input tensor with shape [num_batches, num_classes, spatial_dimension].
                only one class score here.
            detections_per_class (Tensor, optional): Integer representing the maximum number of
                boxes to be selected per batch per class. It is a scalar.
            iou_thresh (Tensor, optional): Float representing the threshold for deciding whether
                boxes overlap too much with respect to IOU. It is scalar. Value range [0, 1].
            score_thresh (Tensor, optional): Float representing the threshold for deciding when to
                remove boxes based on score. It is a scalar.

        Returns:
            Tensor(int64): selected indices from the boxes tensor. [num_selected_indices, 3],
                the selected index format is [batch_index, class_index, box_index].
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
    def symbolic(g, boxes, scores, detections_per_class, iou_thresh, score_thresh):
        return g.op("NonMaxSuppression", boxes, scores, detections_per_class, iou_thresh, score_thresh)
