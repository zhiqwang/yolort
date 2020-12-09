# Modified from ultralytics/yolov5 by Zhiqiang Wang
import torch
from torch import nn, Tensor

from torch.jit.annotations import Tuple, List, Dict, Optional
from torchvision.ops import batched_nms, box_iou

from . import _utils as det_utils
from ._utils import FocalLoss
from utils.box_ops import bbox_iou


def _sum(x: List[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res


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


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


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
        box: float = 0.05,  # box loss gain
        cls: float = 0.5,  # cls loss gain
        cls_pw: float = 1.0,  # cls BCELoss positive_weight
        obj: float = 1.0,  # obj loss gain (scale with pixels)
        obj_pw: float = 1.0,  # obj BCELoss positive_weight
        anchor_t: Tuple[float] = (1.0, 2.0, 8.0),  # anchor-multiple threshold
        gr: float = 1.0,  # iou loss ratio (obj_loss = 1.0 or iou)
        fl_gamma: float = 0.0,  # focal loss gamma
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
        head_outputs: Tensor,
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

            match_quality_matrix = box_iou(targets_per_image['boxes'], anchors)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        return self.compute_loss(targets, head_outputs, anchors, matched_idxs)

    def compute_loss(
        self,
        targets: List[Dict[str, Tensor]],
        head_outputs: Tensor,
        anchors: Tensor,
        matched_idxs: List[Tensor],
    ) -> Dict[str, Tensor]:
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        device = anchors.device
        num_classes = head_outputs.shape[2] - 5
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.cls_pw])).to(device)
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.obj_pw])).to(device)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cp, cn = smooth_BCE(eps=0.0)

        # Focal loss
        g = self.fl_gamma  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # Losses
        num_targets = 0  # number of targets
        num_output = len(head_outputs)  # number of outputs
        balance = [4.0, 1.0, 0.4] if num_output == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
        for i, pi in enumerate(head_outputs):  # layer index, layer predictions
            b, a, gj, gi = matched_idxs[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                num_targets += n  # cumulative targets
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
                iou = bbox_iou(pbox.T, targets['boxes'][i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if num_classes > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                    t[range(n), targets['labels'][i]] = cp
                    lcls += BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

        out_scaling = 3 / num_output  # output count scaling
        lbox *= self.box * out_scaling
        lobj *= self.obj * out_scaling * (1.4 if num_output == 4 else 1.)
        lcls *= self.cls * out_scaling
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()


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
