# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
import math
import torch
from torch import nn, Tensor

from torchvision.ops import boxes as box_ops

from . import _utils as det_utils

from typing import Tuple, List, Dict


class YOLOHead(nn.Module):
    def __init__(self, in_channels: List[int], num_anchors: int, strides: List[int], num_classes: int):
        super().__init__()
        self.num_anchors = num_anchors  # anchors
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5  # number of outputs per anchor
        self.strides = strides

        self.head = nn.ModuleList(
            nn.Conv2d(ch, self.num_outputs * self.num_anchors, 1) for ch in in_channels)  # output conv

        self._initialize_biases()  # Init weights, biases

    def _initialize_biases(self, cf=None):
        """
        Initialize biases into YOLOHead, cf is class frequency
        Check section 3.3 in <https://arxiv.org/abs/1708.02002>
        """
        for mi, s in zip(self.head, self.strides):
            b = mi.bias.view(self.num_anchors, -1)  # conv.bias(255) to (3,85)
            # obj (8 objects per 640 image)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            # classes
            b.data[:, 5:] += torch.log(cf / cf.sum()) if cf else math.log(0.6 / (self.num_classes - 0.99))
            mi.bias = nn.Parameter(b.view(-1), requires_grad=True)

    def get_result_from_head(self, features: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.head[idx](features),
        but torchscript doesn't support this yet
        """
        num_blocks = 0
        for m in self.head:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        i = 0
        out = features
        for module in self.head:
            if i == idx:
                out = module(features)
            i += 1
        return out

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        all_pred_logits = []  # inference output

        for i, features in enumerate(x):
            pred_logits = self.get_result_from_head(features, i)

            # Permute output from (N, A * K, H, W) to (N, A, H, W, K)
            N, _, H, W = pred_logits.shape
            pred_logits = pred_logits.view(N, self.num_anchors, -1, H, W)
            pred_logits = pred_logits.permute(0, 1, 3, 4, 2)  # Size=(N, A, H, W, K)

            all_pred_logits.append(pred_logits)

        return all_pred_logits


class SetCriterion:
    """
    This class computes the loss for YOLOv5.
    """
    def __init__(self, iou_thresh: float = 0.5) -> None:
        """
        Args:
            iou_thresh (float): quality values greater than or equal to
                this value are candidate matches.
        """
        self.proposal_matcher = det_utils.Matcher(iou_threshold=iou_thresh)

    def __call__(
        self,
        targets: Tensor,
        head_outputs: List[Tensor],
        anchors_tuple: Tuple[Tensor, Tensor, Tensor],
    ) -> Dict[str, Tensor]:
        """ This performs the loss computation.
        Parameters:
            targets: list of dicts, such that len(targets) == batch_size.
                    The expected keys in each dict depends on the losses applied, see each loss' doc
            head_outputs: dict of tensors, see the output specification of the model for the format
        """
        matched_idxs = []

        return self.compute_loss(targets, head_outputs, matched_idxs)

    def compute_loss(
        self,
        targets: Tensor,
        head_outputs: List[Tensor],
        matched_idxs: List[Tensor],
    ):
        device = targets.device

        loss_cls = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)

        return {
            'cls_logits': loss_cls,
            'bbox_regression': loss_box,
            'objectness': loss_obj,
        }


class PostProcess(nn.Module):
    """Performs Non-Maximum Suppression (NMS) on inference results"""
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
    }
    def __init__(
        self,
        score_thresh: float,
        nms_thresh: float,
        detections_per_img: int,
    ) -> None:
        """
        Arguments:
            score_thresh (float)
            nms_thresh (float)
            detections_per_img (int)
        """
        super().__init__()
        self.box_coder = det_utils.BoxCoder()
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img  # maximum number of detections per image

    def forward(
        self,
        head_outputs: List[Tensor],
        anchors_tuple: Tuple[Tensor, Tensor, Tensor],
    ) -> List[Dict[str, Tensor]]:
        """ Perform the computation. At test time, postprocess_detections is the final layer of YOLO.
        Decode location preds, apply non-maximum suppression to location predictions based on conf
        scores and threshold to a detections_per_img number of output predictions for both confidence
        score and locations.

        Parameters:
            head_outputs: [batch_size, num_anchors, num_classes + 5] predicted locations and
                class/object confidence.
            anchors_tuple:
        """
        batch_size, _, _, _, K = head_outputs[0].shape

        all_pred_logits = []
        for pred_logits in head_outputs:
            pred_logits = pred_logits.reshape(batch_size, -1, K)  # Size=(NN, HWA, K)
            all_pred_logits.append(pred_logits)

        all_pred_logits = torch.cat(all_pred_logits, dim=1)

        detections: List[Dict[str, Tensor]] = []

        for idx in range(batch_size):  # image idx, image inference
            pred_logits = torch.sigmoid(all_pred_logits[idx])

            # Compute conf
            # box_conf x class_conf, w/ shape: num_anchors x num_classes
            scores = pred_logits[:, 5:] * pred_logits[:, 4:5]

            boxes = self.box_coder.decode_single(pred_logits[:, :4], anchors_tuple)

            # remove low scoring boxes
            inds, labels = torch.where(scores > self.score_thresh)
            boxes, scores = boxes[inds], scores[inds, labels]

            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring head_outputs
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            detections.append({'scores': scores, 'labels': labels, 'boxes': boxes})

        return detections
