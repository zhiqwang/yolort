from typing import Dict, List, Callable, Optional

from torch import nn
from torchvision.models import mobilenet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool
from yolort.utils import load_state_dict_from_url

from .anchor_utils import AnchorGenerator
from .box_head import YOLOHead
from .yolo import YOLO

__all__ = ["yolov5_mobilenet_v3_small_fpn"]


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediateLayerGetter apply here.

    Args:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.

    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(
        self,
        backbone: nn.Module,
        return_layers: Dict[str, str],
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
    ) -> None:
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)

        return list(x.values())  # unpack OrderedDict into two lists for easier handling


def mobilenet_backbone(
    backbone_name: str,
    pretrained: bool,
    norm_layer: Callable[..., nn.Module] = misc_nn_ops.FrozenBatchNorm2d,
    trainable_layers: int = 2,
    returned_layers: Optional[List[int]] = None,
) -> nn.Module:
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
        returned_layers = [num_stages - 3, num_stages - 2, num_stages - 1]
    assert min(returned_layers) >= 0 and max(returned_layers) < num_stages
    return_layers = {f"{stage_indices[k]}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_list = [backbone[stage_indices[i]].out_channels for i in returned_layers]
    return BackboneWithFPN(
        backbone,
        return_layers,
        in_channels_list,
        out_channels,
        extra_blocks=LastLevelMaxPool(),
    )


def _yolov5_mobilenet_v3_small_fpn(
    weights_name: str,
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs,
):
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 6, 3
    )

    if pretrained:
        pretrained_backbone = False
    backbone = mobilenet_backbone(
        "mobilenet_v3_small",
        pretrained_backbone,
        trainable_layers=trainable_backbone_layers,
    )
    strides = [8, 16, 32, 64]
    anchor_grids = [
        [19, 27, 44, 40, 38, 94],
        [96, 68, 86, 152, 180, 137],
        [140, 301, 303, 264, 238, 542],
        [436, 615, 739, 380, 925, 792],
    ]
    anchor_generator = AnchorGenerator(strides, anchor_grids)
    head = YOLOHead(
        backbone.out_channels,
        anchor_generator.num_anchors,
        anchor_generator.strides,
        num_classes,
    )
    model = YOLO(backbone, num_classes, anchor_generator=anchor_generator, head=head, **kwargs)
    if pretrained:
        if model_urls.get(weights_name, None) is None:
            raise ValueError(f"No checkpoint is available for model {weights_name}")
        state_dict = load_state_dict_from_url(model_urls[weights_name], progress=progress)
        model.load_state_dict(state_dict)
    return model


model_urls = {
    "yolov5_mobilenet_v3_small_fpn_coco": None,
}


def yolov5_mobilenet_v3_small_fpn(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 80,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs,
):
    """
    Constructs a high resolution YOLOv5 model with a MobileNetV3-Large FPN backbone.
    It works similarly to Faster R-CNN with ResNet-50 FPN backbone. See
    :func:`~torchvision.models.detection.fasterrcnn_resnet50_fpn` for more
    details.

    Note:
        We do not provide a pre-trained model with mobilenet as the backbone now, this function
        is just used as an example of how to construct a YOLOv5 model with TorchVision's pre-trained
        MobileNetV3-Small FPN backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting
            from final block. Valid values are between 0 and 6, with 6 meaning all backbone layers
            are trainable.
    """
    weights_name = "yolov5_mobilenet_v3_small_fpn_coco"

    return _yolov5_mobilenet_v3_small_fpn(
        weights_name,
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        **kwargs,
    )
