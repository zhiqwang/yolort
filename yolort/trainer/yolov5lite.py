from typing import List, Optional

from torch import nn, Tensor
from torchvision.models import mobilenet
from torchvision.models.detection.backbone_utils import _validate_trainable_layers, BackboneWithFPN
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from yolort.models.anchor_utils import AnchorGenerator
from yolort.models.box_head import YOLOHead

__all__ = ["yolov5lite"]


class YOLOv5Lite(nn.Module):
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
    ):
        super().__init__()

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

        out_channels = [256, 256, 256]

        if head is None:
            head = YOLOHead(
                out_channels,
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
        # unpack OrderedDict into two lists for easier handling
        features = list(features.values())

        # compute the yolo heads outputs using the features
        head_outputs = self.head(features)

        return head_outputs


def mobilenet_backbone(
    backbone_name,
    pretrained,
    norm_layer=misc_nn_ops.FrozenBatchNorm2d,
    trainable_layers=2,
    returned_layers=None,
):
    backbone = mobilenet.__dict__[backbone_name](pretrained=pretrained, norm_layer=norm_layer).features

    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = (
        [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    )
    num_stages = len(stage_indices)

    # find the index of the layer from which we wont freeze
    assert 0 <= trainable_layers <= num_stages
    freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    out_channels = 256

    if returned_layers is None:
        returned_layers = [num_stages - 2, num_stages - 1]
    assert min(returned_layers) >= 0 and max(returned_layers) < num_stages
    return_layers = {f'{stage_indices[k]}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_list = [backbone[stage_indices[i]].out_channels for i in returned_layers]
    return BackboneWithFPN(
        backbone,
        return_layers,
        in_channels_list,
        out_channels,
        extra_blocks=LastLevelMaxPool(),
    )


def _yolov5_mobilenet_v3_large_fpn(
    weights_name,
    pretrained=False,
    progress=True,
    num_classes=80,
    pretrained_backbone=True,
    trainable_backbone_layers=None,
    **kwargs,
):
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 6, 3
    )

    if pretrained:
        pretrained_backbone = False
    backbone = mobilenet_backbone(
        "mobilenet_v3_large",
        pretrained_backbone,
        trainable_layers=trainable_backbone_layers,
    )

    model = YOLOv5Lite(backbone, num_classes, **kwargs)

    return model


def yolov5lite(
    pretrained=False,
    progress=True,
    num_classes=80,
    pretrained_backbone=True,
    trainable_backbone_layers=None,
    **kwargs,
):
    """
    Constructs a high resolution Faster R-CNN model with a MobileNetV3-Large FPN backbone.
    It works similarly to Faster R-CNN with ResNet-50 FPN backbone. See
    :func:`~torchvision.models.detection.fasterrcnn_resnet50_fpn` for more
    details.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting
            from final block. Valid values are between 0 and 6, with 6 meaning all backbone layers
            are trainable.
    """
    weights_name = "yolov5_mobilenet_v3_large_fpn_coco"

    return _yolov5_mobilenet_v3_large_fpn(
        weights_name,
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        **kwargs,
    )
