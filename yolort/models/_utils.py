# Copyright (c) 2020, yolort team. All rights reserved.

import math
from functools import reduce
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
from yolort.v5 import get_yolov5_size, load_yolov5_model

from . import yolo


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


def load_from_ultralytics(checkpoint_path: str, version: str = "r6.0"):
    """
    Allows the user to load model state file from the checkpoint trained from
    the ultralytics/yolov5.

    Args:
        checkpoint_path (str): Path of the YOLOv5 checkpoint model.
        version (str): upstream version released by the ultralytics/yolov5, Possible
            values are ["r3.1", "r4.0", "r6.0"]. Default: "r6.0".
    """

    assert version in ["r3.1", "r4.0", "r6.0"], "Currently does not support this version."

    checkpoint_yolov5 = load_yolov5_model(checkpoint_path)
    num_classes = checkpoint_yolov5.yaml["nc"]
    strides = checkpoint_yolov5.stride
    # YOLOv5 will change the anchors setting when using the auto-anchor mechanism. So we
    # use the following formula to compute the anchor_grids instead of attaching it via
    # checkpoint_yolov5.yaml["anchors"]
    num_anchors = checkpoint_yolov5.model[-1].anchors.shape[1]
    anchor_grids = (
        (checkpoint_yolov5.model[-1].anchors * checkpoint_yolov5.model[-1].stride.view(-1, 1, 1))
        .reshape(1, -1, 2 * num_anchors)
        .tolist()[0]
    )

    depth_multiple = checkpoint_yolov5.yaml["depth_multiple"]
    width_multiple = checkpoint_yolov5.yaml["width_multiple"]

    use_p6 = False
    if len(strides) == 4:
        use_p6 = True

    if use_p6:
        inner_block_maps = {"0": "11", "1": "12", "3": "15", "4": "16", "6": "19", "7": "20"}
        layer_block_maps = {"0": "23", "1": "24", "2": "26", "3": "27", "4": "29", "5": "30", "6": "32"}
        p6_block_maps = {"0": "9", "1": "10"}
        head_ind = 33
        head_name = "m"
    else:
        inner_block_maps = {"0": "9", "1": "10", "3": "13", "4": "14"}
        layer_block_maps = {"0": "17", "1": "18", "2": "20", "3": "21", "4": "23"}
        p6_block_maps = None
        head_ind = 24
        head_name = "m"

    module_state_updater = ModuleStateUpdate(
        depth_multiple,
        width_multiple,
        inner_block_maps=inner_block_maps,
        layer_block_maps=layer_block_maps,
        p6_block_maps=p6_block_maps,
        strides=strides,
        anchor_grids=anchor_grids,
        head_ind=head_ind,
        head_name=head_name,
        num_classes=num_classes,
        version=version,
        use_p6=use_p6,
    )
    module_state_updater.updating(checkpoint_yolov5)
    state_dict = module_state_updater.model.half().state_dict()

    size = get_yolov5_size(depth_multiple, width_multiple)

    return {
        "num_classes": num_classes,
        "depth_multiple": depth_multiple,
        "width_multiple": width_multiple,
        "strides": strides,
        "anchor_grids": anchor_grids,
        "use_p6": use_p6,
        "size": size,
        "state_dict": state_dict,
    }


class ModuleStateUpdate:
    """
    Update checkpoint from ultralytics yolov5.
    """

    def __init__(
        self,
        depth_multiple: float,
        width_multiple: float,
        inner_block_maps: Optional[Dict[str, str]] = None,
        layer_block_maps: Optional[Dict[str, str]] = None,
        p6_block_maps: Optional[Dict[str, str]] = None,
        strides: Optional[List[int]] = None,
        anchor_grids: Optional[List[List[float]]] = None,
        head_ind: int = 24,
        head_name: str = "m",
        num_classes: int = 80,
        version: str = "r6.0",
        use_p6: bool = False,
    ) -> None:

        # Configuration for making the keys consistent
        if inner_block_maps is None:
            inner_block_maps = {"0": "9", "1": "10", "3": "13", "4": "14"}
        self.inner_block_maps = inner_block_maps
        if layer_block_maps is None:
            layer_block_maps = {"0": "17", "1": "18", "2": "20", "3": "21", "4": "23"}
        self.layer_block_maps = layer_block_maps
        self.p6_block_maps = p6_block_maps
        self.head_ind = head_ind
        self.head_name = head_name

        # Set model
        yolov5_size = get_yolov5_size(depth_multiple, width_multiple)
        backbone_name = f"darknet_{yolov5_size}_{version.replace('.', '_')}"
        self.model = yolo.build_model(
            backbone_name,
            depth_multiple,
            width_multiple,
            version,
            num_classes=num_classes,
            use_p6=use_p6,
            strides=strides,
            anchor_grids=anchor_grids,
        )

    def updating(self, state_dict):
        # Obtain module state
        state_dict = obtain_module_sequential(state_dict)

        # Update backbone weights
        for name, params in self.model.backbone.body.named_parameters():
            params.data.copy_(self.attach_parameters_block(state_dict, name, None))

        for name, buffers in self.model.backbone.body.named_buffers():
            buffers.copy_(self.attach_parameters_block(state_dict, name, None))

        # Update PAN weights
        # Updating P6 weights
        if self.p6_block_maps is not None:
            for name, params in self.model.backbone.pan.intermediate_blocks.p6.named_parameters():
                params.data.copy_(self.attach_parameters_block(state_dict, name, self.p6_block_maps))

            for name, buffers in self.model.backbone.pan.intermediate_blocks.p6.named_buffers():
                buffers.copy_(self.attach_parameters_block(state_dict, name, self.p6_block_maps))

        # Updating inner_block weights
        for name, params in self.model.backbone.pan.inner_blocks.named_parameters():
            params.data.copy_(self.attach_parameters_block(state_dict, name, self.inner_block_maps))

        for name, buffers in self.model.backbone.pan.inner_blocks.named_buffers():
            buffers.copy_(self.attach_parameters_block(state_dict, name, self.inner_block_maps))

        # Updating layer_block weights
        for name, params in self.model.backbone.pan.layer_blocks.named_parameters():
            params.data.copy_(self.attach_parameters_block(state_dict, name, self.layer_block_maps))

        for name, buffers in self.model.backbone.pan.layer_blocks.named_buffers():
            buffers.copy_(self.attach_parameters_block(state_dict, name, self.layer_block_maps))

        # Update YOLOHead weights
        for name, params in self.model.head.named_parameters():
            params.data.copy_(self.attach_parameters_heads(state_dict, name))

        for name, buffers in self.model.head.named_buffers():
            buffers.copy_(self.attach_parameters_heads(state_dict, name))

    @staticmethod
    def attach_parameters_block(state_dict, name, block_maps=None):
        keys = name.split(".")
        ind = int(block_maps[keys[0]]) if block_maps else int(keys[0])
        return rgetattr(state_dict[ind], keys[1:])

    def attach_parameters_heads(self, state_dict, name):
        keys = name.split(".")
        ind = int(keys[1])
        return rgetattr(getattr(state_dict[self.head_ind], self.head_name)[ind], keys[2:])


def rgetattr(obj, attr, *args):
    """
    Nested version of getattr.
    Ref: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return reduce(_getattr, [obj] + attr)


def obtain_module_sequential(state_dict):
    if isinstance(state_dict, nn.Sequential):
        return state_dict
    else:
        return obtain_module_sequential(state_dict.model)


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
