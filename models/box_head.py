# Modified from ultralytics/yolov5 by Zhiqiang Wang
import torch
from torch import nn, Tensor

from torch.jit.annotations import Tuple, List, Dict, Optional
from torchvision.ops import batched_nms, box_iou

from . import _utils as det_utils


def _sum(x: List[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res


class YoloHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):  # detection layer
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

    def forward(self, x: List[Tensor]) -> Tensor:
        all_pred_logits = torch.jit.annotate(List[Tensor], [])  # inference output

        for i, features in enumerate(x):
            pred_logits = self.get_result_from_head(features, i)

            # Permute output from (N, A * K, H, W) to (N, HWA, K)
            N, _, H, W = pred_logits.shape
            pred_logits = pred_logits.view(N, self.num_anchors, -1, H, W)
            pred_logits = pred_logits.permute(0, 1, 3, 4, 2)
            pred_logits = pred_logits.reshape(N, -1, self.num_outputs)  # Size=(N, HWA, K)

            all_pred_logits.append(pred_logits)

        return torch.cat(all_pred_logits, dim=1)


class SetCriterion(nn.Module):
    """This class computes the loss for YOLOv5.
    Arguments:
        variances:
    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
    }

    def __init__(
        self,
        weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        fg_iou_thresh: float = 0.5,
        bg_iou_thresh: float = 0.4,
        allow_low_quality_matches: bool = True,
    ) -> None:
        """
        Arguments:
            weights (4-element tuple)
            fg_iou_thresh (float)
            bg_iou_thresh (float)
            allow_low_quality_matches (bool)
        """
        super().__init__()

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=allow_low_quality_matches,
        )

        self.box_coder = det_utils.BoxCoder(weights=weights)

    def forward(
        self,
        targets: List[Dict[str, Tensor]],
        bbox_regression: Tensor,
        anchors: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Arguments:
            targets (List[Dict[Tensor]]): ground-truth boxes present in the image
            head_outputs (Dict[Tensor])
            anchor (List[Tensor])
        """
        matched_idxs = []
        for targets_per_image in targets:
            if targets_per_image['boxes'].numel() == 0:
                matched_idxs.append(torch.full((anchors.size(0),), -1, dtype=torch.int64))
                continue

            match_quality_matrix = box_iou(targets_per_image['boxes'], anchors)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        return self.compute_loss(targets, bbox_regression, anchors, matched_idxs)

    def compute_loss(
        self,
        targets: List[Dict[str, Tensor]],
        bbox_regression: Tensor,
        anchors: Tensor,
        matched_idxs: List[Tensor],
    ) -> Dict[str, Tensor]:
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = []

        for targets_per_image, bbox_regression_per_image, matched_idxs_per_image in zip(
                targets, bbox_regression, matched_idxs):
            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()

            # select only the foreground boxes
            matched_gt_boxes_per_image = targets_per_image['boxes'][matched_idxs_per_image[foreground_idxs_per_image]]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors = anchors[foreground_idxs_per_image, :]

            # compute the regression targets
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors)

            # compute the loss
            losses.append(torch.nn.functional.l1_loss(
                bbox_regression_per_image,
                target_regression,
                size_average=False
            ) / max(1, num_foreground))

        return {
            'loss': _sum(losses) / max(1, len(targets)),
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
        head_outputs: Tensor,
        anchors_tuple: Tuple[Tensor, Tensor, Tensor],
        image_shapes: Optional[List[Tuple[int, int]]] = None,
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
        num_images = len(image_shapes)
        detections = torch.jit.annotate(List[Dict[str, Tensor]], [])

        for index in range(num_images):  # image index, image inference
            pred_logits = torch.sigmoid(head_outputs[index])

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
