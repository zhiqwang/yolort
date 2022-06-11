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


class EfficientNMSOp(torch.autograd.Function):
    """
    The NMS algorithm in this plugin first filters the scores below the given
    ``scoreThreshold``. This subset of scores is then sorted, and their corresponding
    boxes are then further filtered out by removing boxes that overlap each other with
    an IOU above the given ``iouThreshold``.

    Most object detection networks work by generating raw predictions from a
    "localization head" which adjust the coordinates of standard non-learned anchor
    coordinates to produce a tighter fitting bounding box. This process is called
    "box decoding", and it usually involves a large number of element-wise operations
    to transform the anchors to final box coordinates. As this can involve exponential
    operations on a large number of anchors, it can be computationally expensive, so
    this plugin gives the option of fusing the box decoder within the NMS operation
    which can be done in a far more efficient manner, resulting in lower latency for
    the network.

    Reference:
        https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin#efficient-nms-plugin
    """

    @staticmethod
    def forward(
        ctx,
        boxes: Tensor,
        scores: Tensor,
        background_class: int = -1,
        box_coding: int = 0,
        iou_thresh: float = 0.45,
        score_thresh: float = 0.35,
        max_output_boxes: int = 100,
        plugin_version: str = "1",
        score_activation: int = 0,
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
        iou_thresh: float = 0.45,
        score_thresh: float = 0.35,
        max_output_boxes: int = 100,
        plugin_version: str = "1",
        score_activation: int = 0,
    ):

        return g.op(
            "TRT::EfficientNMS_TRT",
            boxes,
            scores,
            background_class_i=background_class,
            box_coding_i=box_coding,
            iou_threshold_f=iou_thresh,
            score_threshold_f=score_thresh,
            max_output_boxes_i=max_output_boxes,
            plugin_version_s=plugin_version,
            score_activation_i=score_activation,
            outputs=4,
        )
