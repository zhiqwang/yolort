# Copyright (c) 2022, yolort team. All rights reserved.

from typing import Any, List, Dict, Callable, Optional

from torch import nn, Tensor
from yolort.models.anchor_utils import AnchorGenerator
from yolort.models.backbone_utils import darknet_pan_backbone
from yolort.models.box_head import YOLOHead


class YOLOTraining(nn.Module):
    """
    Another way to implement ultralytics/yolov5 models with mobilenetv3 for training.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        # Anchor parameters
        strides: Optional[List[int]] = None,
        anchor_grids: Optional[List[List[float]]] = None,
        anchor_generator: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        # Training parameter
        criterion: Optional[Callable[..., Dict[str, Tensor]]] = None,
    ):
        super().__init__()
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )
        self.backbone = backbone

        if strides is None:
            strides: List[int] = [8, 16, 32]

        if anchor_grids is None:
            anchor_grids: List[List[float]] = [
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326],
            ]

        if anchor_generator is None:
            anchor_generator = AnchorGenerator(strides, anchor_grids)
        self.anchor_generator = anchor_generator

        self.compute_loss = criterion

        if head is None:
            head = YOLOHead(
                backbone.out_channels,
                anchor_generator.num_anchors,
                anchor_generator.strides,
                num_classes,
            )
        self.head = head

    def forward(self, samples: Tensor):
        """
        Args:
            samples (NestedTensor): Expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        # get the features from the backbone
        features = self.backbone(samples)

        # compute the yolo heads outputs using the features
        head_outputs = self.head(features)

        return head_outputs


def build_model(
    backbone_name: str,
    depth_multiple: float,
    width_multiple: float,
    version: str,
    weights_name: Optional[str] = None,
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    use_p6: bool = False,
    **kwargs: Any,
) -> YOLOTraining:
    """
    Constructs a YOLO model.

    Example::

        >>> model = yolov5(pretrained=True)
        >>> model.eval()
        >>> x = torch.rand(4, 3, 416, 320)
        >>> predictions = model(x)

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        version (str): Module version released by ultralytics. Possible values
            are ["r3.1", "r4.0", "r6.0"].
    """
    backbone = darknet_pan_backbone(
        backbone_name, depth_multiple, width_multiple, version=version, use_p6=use_p6
    )

    model = YOLOTraining(backbone, num_classes, **kwargs)

    return model


def yolov5n(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    **kwargs: Any,
) -> YOLOTraining:
    r"""yolov5 nano release 6.0 model from"""
    backbone_name = "darknet_n_r6_0"
    weights_name = "yolov5_darknet_pan_n_r60_coco"
    depth_multiple = 0.33
    width_multiple = 0.25
    version = "r6.0"
    return build_model(
        backbone_name,
        depth_multiple,
        width_multiple,
        version,
        weights_name,
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        **kwargs,
    )
