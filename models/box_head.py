# Modified from ultralytics/yolov5 by Zhiqiang Wang
import torch
from torch import nn, Tensor

from torchvision.ops import batched_nms

from . import _utils as det_utils

from typing import Tuple, List, Dict


class YoloHead(nn.Module):
    def __init__(self, in_channels: List[int], num_anchors: int, num_classes: int):
        super().__init__()
        self.num_anchors = num_anchors  # anchors
        self.num_outputs = num_classes + 5  # number of outputs per anchor

        self.head = nn.ModuleList(
            nn.Conv2d(ch, self.num_outputs * self.num_anchors, 1) for ch in in_channels)  # output conv

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
        all_pred_logits: List[Tensor] = []  # inference output

        for i, features in enumerate(x):
            pred_logits = self.get_result_from_head(features, i)

            # Permute output from (N, A * K, H, W) to (N, A, H, W, K)
            N, _, H, W = pred_logits.shape
            pred_logits = pred_logits.view(N, self.num_anchors, -1, H, W)
            pred_logits = pred_logits.permute(0, 1, 3, 4, 2)  # Size=(N, A, H, W, K)

            all_pred_logits.append(pred_logits)

        return all_pred_logits


class SetCriterion(nn.Module):
    """This class computes the loss for YOLOv5.
    Arguments:
        variances:
    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
    }

    def __init__(
        self,
        strides: List[int],
        anchor_grids: List[List[int]],
        box: float = 0.05,  # box loss gain
        cls: float = 0.5,  # cls loss gain
        cls_pw: float = 1.0,  # cls BCELoss positive_weight
        obj: float = 1.0,  # obj loss gain (scale with pixels)
        obj_pw: float = 1.0,  # obj BCELoss positive_weight
        anchor_threshold: float = 4.0,  # anchor-multiple threshold
        iou_ratio: float = 1.0,  # iou loss ratio (obj_loss = 1.0 or iou)
        fl_gamma: float = 0.0,  # focal loss gamma
        layer_balance: List[float] = [4.0, 1.0, 0.4],
    ) -> None:
        """
        Arguments:
            weights (4-element tuple)
            fg_iou_thresh (float)
            bg_iou_thresh (float)
            allow_low_quality_matches (bool)
        """
        super().__init__()
        self.strides = strides
        self.anchor_grids = anchor_grids
        self.anchor_threshold = anchor_threshold
        self.fl_gamma = fl_gamma
        self.layer_balance = layer_balance

        self.cls_pw = cls_pw
        self.obj_pw = obj_pw
        self.cls = cls
        self.obj = obj
        self.box = box

        self.iou_ratio = iou_ratio

    def forward(
        self,
        head_outputs: List[Tensor],
        targets: Tensor,
    ) -> Dict[str, Tensor]:
        """ This performs the loss computation.
        Parameters:
            head_outputs: dict of tensors, see the output specification of the model for the format
            targets: list of dicts, such that len(targets) == batch_size.
                    The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        targets_cls, targets_box, indices, anchors = self.select_training_samples(head_outputs, targets)
        losses = self.compute_loss(head_outputs, targets_cls, targets_box, indices, anchors)

        return losses

    def select_training_samples(
        self,
        head_outputs: List[Tensor],
        targets: Tensor,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tuple[Tensor, Tensor, Tensor, Tensor]], List[Tensor]]:
        # get boxes indices for each anchors
        device = head_outputs[0].device

        num_layers = len(head_outputs)
        anchors = torch.as_tensor(self.anchor_grids, dtype=torch.float32, device=device)
        strides = torch.as_tensor(self.strides, dtype=torch.float32, device=device)
        anchors = anchors.view(num_layers, -1, 2) / strides.view(-1, 1, 1)
        targets_cls, targets_box, indices, anchors_encode = self.assign_targets_to_anchors(
            head_outputs, anchors, targets)

        return targets_cls, targets_box, indices, anchors_encode

    def assign_targets_to_anchors(
        self,
        head_outputs: List[Tensor],
        anchors: Tensor,
        targets: Tensor,
    ):
        """Assign ground truth boxes and targets to anchors.
        Args:
            gt_boxes (List[Tensor]): with shape num_targets x 4, ground truth boxes
            gt_labels (List[Tensor]): with shape num_targets, labels of targets
            anchors (Tensor): with shape num_priors x 4, XYXY_REL BoxMode
        Returns:
            boxes (List[Tensor]): with shape num_priors x 4 real values for anchors.
            labels (List[Tensor]): with shape num_priros, labels for anchors.
        """
        device = head_outputs[0].device
        num_layers = len(head_outputs)
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        num_anchors = len(self.anchor_grids)  # number of anchors
        num_targets = len(targets)  # number of targets

        targets_cls, targets_box, anchors_encode = [], [], []
        indices: List[Tuple[Tensor, Tensor, Tensor, Tensor]] = []
        gain = torch.ones(7, device=device)  # normalized to gridspace gain
        # same as .repeat_interleave(num_targets)
        ai = torch.arange(num_anchors, device=device).float().view(num_anchors, 1).repeat(1, num_targets)
        targets = torch.cat((targets.repeat(num_anchors, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=device).float() * g  # offsets

        for i in range(num_layers):
            anchors_per_layer = anchors[i]
            gain[2:6] = torch.tensor(head_outputs[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            targets_with_gain = targets * gain
            if num_targets:
                # Matches
                ratios_wh = targets_with_gain[:, :, 4:6] / anchors_per_layer[:, None]  # wh ratio
                ratios_filtering = torch.max(ratios_wh, 1. / ratios_wh).max(2)[0]
                inds = torch.where(ratios_filtering < self.anchor_threshold)
                targets_with_gain = targets_with_gain[inds]  # filter

                # Offsets
                grid_xy = targets_with_gain[:, 2:4]  # grid xy
                grid_xy_inverse = gain[[2, 3]] - grid_xy  # inverse
                inds_jk = (grid_xy % 1. < g) & (grid_xy > 1.)
                inds_lm = (grid_xy_inverse % 1. < g) & (grid_xy_inverse > 1.)
                inds_ones = torch.ones_like(inds_jk[:, 0])[:, None]
                inds = torch.cat((inds_ones, inds_jk, inds_lm), dim=1).T
                targets_with_gain = targets_with_gain.repeat((5, 1, 1))[inds]
                offsets = (torch.zeros_like(grid_xy)[None] + off[:, None])[inds]
            else:
                targets_with_gain = targets[0]
                offsets = torch.tensor(0, device=device)

            # Define
            bc = targets_with_gain[:, :2].long().T  # image, class
            grid_xy = targets_with_gain[:, 2:4]  # grid xy
            grid_wh = targets_with_gain[:, 4:6]  # grid wh
            grid_ij = (grid_xy - offsets).long()

            # Append
            a = targets_with_gain[:, 6].long()  # anchor indices
            # image, anchor, grid indices
            indices.append((bc[0], a, grid_ij[:, 1].clamp_(0, gain[3] - 1), grid_ij[:, 0].clamp_(0, gain[2] - 1)))
            targets_box.append(torch.cat((grid_xy - grid_ij, grid_wh), 1))  # box
            anchors_encode.append(anchors_per_layer[a])  # anchors
            targets_cls.append(bc[1])  # class

        return targets_cls, targets_box, indices, anchors_encode

    @staticmethod
    def label_smooth_bce(eps: float = 0.1):
        '''
        Return positive, negative label smoothing BCE targets
        <https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441>
        '''
        return 1.0 - 0.5 * eps, 0.5 * eps

    def compute_loss(
        self,
        head_outputs: List[Tensor],
        targets_cls: List[Tensor],
        targets_box: List[Tensor],
        matched_idxs: List[Tuple[Tensor, Tensor, Tensor, Tensor]],
        anchors: List[Tensor],
    ) -> Dict[str, Tensor]:
        """ This performs the loss computation.
        Parameters:
            outputs: dict of tensors, see the output specification of the model for the format
            targets: list of dicts, such that len(targets) == batch_size.
                The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        device = head_outputs[0].device
        num_classes = head_outputs[0].shape[-1] - 5
        loss_cls = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cls_positive, cls_negative = self.label_smooth_bce(eps=0.0)

        assert len(head_outputs) == len(self.layer_balance)
        # Losses
        num_targets = 0  # number of targets
        num_output = len(head_outputs)  # number of outputs

        cls_pw = torch.tensor([self.cls_pw], device=device)
        obj_pw = torch.tensor([self.obj_pw], device=device)

        for i, pred_logits_per_layer in enumerate(head_outputs):  # layer index, layer predictions
            b, a, gj, gi = matched_idxs[i]  # image, anchor, gridy, gridx
            obj_logits = torch.zeros_like(pred_logits_per_layer[..., 0], device=device)  # target obj

            num_target_per_layer = b.shape[0]  # number of targets
            if num_target_per_layer:
                num_targets += num_target_per_layer  # cumulative targets
                pred_logits_matched = pred_logits_per_layer[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression head
                bbox_xy = pred_logits_matched[:, :2].sigmoid() * 2. - 0.5
                bbox_wh = (pred_logits_matched[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                bbox_regression = torch.cat((bbox_xy, bbox_wh), 1).to(device)  # predicted box
                ciou = det_utils.bbox_ciou(bbox_regression.T, targets_box[i])
                loss_box += (1.0 - ciou).mean()  # iou loss

                # Objectness head
                # iou ratio
                ciou_vals = torch.tensor(ciou.detach().clamp(0), dtype=obj_logits.dtype)
                obj_logits[b, a, gj, gi] = (1.0 - self.iou_ratio) + (self.iou_ratio * ciou_vals)

                # Classification head
                if num_classes > 1:  # cls loss (only if multiple classes)
                    cls_logits = torch.full_like(pred_logits_matched[:, 5:], cls_negative, device=device)  # targets
                    cls_logits[torch.arange(num_target_per_layer), targets_cls[i]] = cls_positive

                    loss_cls += det_utils.cls_loss(pred_logits_matched[:, 5:], cls_logits, pos_weight=cls_pw)  # BCE

            loss_obj += det_utils.obj_loss(
                pred_logits_per_layer[..., 4],
                obj_logits,
                pos_weight=obj_pw,
            ) * self.layer_balance[i]  # obj loss

        out_scaling = 3 / num_output  # output count scaling
        loss_box *= self.box * out_scaling
        loss_obj *= self.obj * out_scaling * (1.4 if num_output == 4 else 1.)
        loss_cls *= self.cls * out_scaling

        return {
            'cls_logits': loss_cls,
            'bbox_regression': loss_box,
            'objectness': loss_obj,
        }


class PostProcess(nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
    }
    """Performs Non-Maximum Suppression (NMS) on inference results"""
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
            head_outputs : [batch_size, num_anchors, num_classes + 5] predicted locations and class/object confidence.
            image_shapes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        batch_size, _, _, _, K = head_outputs[0].shape

        all_pred_logits: List[Tensor] = []
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
            keep = batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring head_outputs
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            detections.append({'scores': scores, 'labels': labels, 'boxes': boxes})

        return detections
