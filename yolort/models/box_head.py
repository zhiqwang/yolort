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

    Args:
        num_anchors (int): The number of anchors.
        num_classes (int): The number of output classes of the model.
        fl_gamma (float): focal loss gamma (efficientDet default gamma=1.5). Default: 0.0.
        box_gain (float): box loss gain. Default: 0.05.
        cls_gain (float): class loss gain. Default: 0.5.
        cls_pos (float): cls BCELoss positive_weight. Default: 1.0.
        obj_gain (float): obj loss gain (scale with pixels). Default: 1.0.
        obj_pos (float): obj BCELoss positive_weight. Default: 1.0.
        anchor_thresh (float): anchor-multiple threshold. Default: 4.0.
        label_smoothing (float): Label smoothing epsilon. Default: 0.0.
        auto_balance (bool): Auto balance. Default: False.
    """
    def __init__(
        self,
        num_anchors: int,
        strides: List[int],
        anchor_grids: List[List[float]],
        num_classes: int,
        fl_gamma: float = 0.0,
        box_gain: float = 0.05,
        cls_gain: float = 0.5,
        cls_pos: float = 1.0,
        obj_gain: float = 1.0,
        obj_pos: float = 1.0,
        anchor_thresh: float = 4.0,
        label_smoothing: float = 0.0,
        auto_balance: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        assert len(strides) == len(anchor_grids)

        self.num_anchors = num_anchors
        self.num_classes = num_classes

        anchors = torch.as_tensor(anchor_grids, dtype=torch.float32, device=device).view(num_anchors, -1, 2)
        strides = torch.as_tensor(strides, dtype=torch.float32, device=device).view(-1, 1, 1)
        self.anchors = anchors / strides

        self.balance = [4.0, 1.0, 0.4]
        self.ssi = 0  # stride 16 index
        self.box_coder = det_utils.BoxCoder()

        self.sort_obj_iou = False

        # Define criteria
        pos_weight_cls = torch.as_tensor([cls_pos], device=device)
        pos_weight_obj = torch.as_tensor([obj_pos], device=device)
        BCE_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight_cls)
        BCE_obj = nn.BCEWithLogitsLoss(pos_weight=pos_weight_obj)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = det_utils.smooth_BCE(eps=label_smoothing)  # positive, negative BCE targets

        # Focal loss
        if fl_gamma > 0:
            BCE_cls = det_utils.FocalLoss(BCE_cls, fl_gamma)
            BCE_obj = det_utils.FocalLoss(BCE_obj, fl_gamma)

        self.BCE_cls, self.BCE_obj = BCE_cls, BCE_obj

        # Parameters for training
        self.gr = 1.0
        self.auto_balance = auto_balance
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.obj_gain = obj_gain
        self.anchor_thresh = anchor_thresh

    def __call__(
        self,
        targets: Tensor,
        head_outputs: List[Tensor],
    ) -> Dict[str, Tensor]:
        """
        This performs the loss computation.

        Args:
            targets (Tensor): list of dicts, such that len(targets) == batch_size. The
                expected keys in each dict depends on the losses applied, see each loss' doc
            head_outputs (List[Tensor]): dict of tensors, see the output specification
                of the model for the format
        """
        target_cls, target_box, indices, anchors = self.build_targets(targets, head_outputs)

        device = targets.device
        loss_cls = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)

        # Computing the losses
        for i, pred_logits in enumerate(head_outputs):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            target_obj = torch.zeros_like(pred_logits[..., 0], device=device)  # target obj

            num_targets = b.shape[0]  # number of targets
            if num_targets > 0:
                # prediction subset corresponding to targets
                pred_logits_subset = pred_logits[b, a, gj, gi]

                # Regression
                pred_box = self.box_coder.encode_single(pred_logits_subset, anchors[i])
                iou = det_utils.bbox_iou(pred_box.T, target_box[i], x1y1x2y2=False, CIoU=True)
                loss_box += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(target_obj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id]
                    score_iou = score_iou[sort_id]
                target_obj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.num_classes > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pred_logits_subset[:, 5:], self.cn, device=device)  # targets
                    t[range(num_targets), target_cls[i]] = self.cp
                    loss_cls += self.BCE_cls(pred_logits_subset[:, 5:], t)  # BCE

            obji = self.BCE_obj(pred_logits[..., 4], target_obj)
            loss_obj += obji * self.balance[i]  # obj loss
            if self.auto_balance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.auto_balance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        loss_box *= self.box_gain
        loss_obj *= self.obj_gain
        loss_cls *= self.cls_gain

        return {
            'cls_logits': loss_cls,
            'bbox_regression': loss_box,
            'objectness': loss_obj,
        }

    def build_targets(self, targets, head_outputs):
        device = targets.device
        num_layers = len(head_outputs)
        num_anchors = self.num_anchors
        num_targets = targets.shape[0]

        gain = torch.ones(7, device=device)  # normalized to gridspace gain
        # same as .repeat_interleave(num_targets)
        ai = torch.arange(num_anchors, device=device).float().view(num_anchors, 1).repeat(1, num_targets)
        # append anchor indices
        targets = torch.cat((targets.repeat(num_anchors, 1, 1), ai[:, :, None]), 2)

        g_bias = 0.5
        offset = torch.tensor([[0, 0],
                               [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                               # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                               ], device=device).float() * g_bias  # offsets

        target_cls, target_box, indices, anch = [], [], [], []

        for i in range(num_layers):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(head_outputs[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if num_targets > 0:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.anchor_thresh  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']
                # iou(3, n) = wh_iou(anchors(3, 2), gwh(n, 2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g_bias) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g_bias) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + offset[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            # image, anchor, grid indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
            target_box.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            target_cls.append(c)  # class

        return target_cls, target_box, indices, anch


class PostProcess(nn.Module):
    """
    Performs Non-Maximum Suppression (NMS) on inference results
    """
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
        Args:
            score_thresh (float): score_thresh (float): Score threshold used for postprocessing the detections.
            nms_thresh (float): nms_thresh (float): NMS threshold used for postprocessing the detections.
            detections_per_img (int): Number of best detections to keep after NMS.
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
        """
        Perform the computation. At test time, postprocess_detections is the final layer of YOLO.
        Decode location preds, apply non-maximum suppression to location predictions based on conf
        scores and threshold to a detections_per_img number of output predictions for both confidence
        score and locations.

        Args:
            head_outputs (List[Tensor]): The predicted locations and class/object confidence,
                shape of the element is (N, A, H, W, K).
            anchors_tuple (Tuple[Tensor, Tensor, Tensor]):
        """
        batch_size, _, _, _, K = head_outputs[0].shape

        all_pred_logits = []
        for pred_logits in head_outputs:
            pred_logits = pred_logits.reshape(batch_size, -1, K)  # Size=(N, HWA, K)
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
