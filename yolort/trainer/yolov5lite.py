import torch

from collections import OrderedDict
from functools import partial
from torch import nn, Tensor
from typing import Any, Callable, Dict, List, Optional

from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models import mobilenet
from torchvision.ops.misc import ConvNormActivation
from yolort.models.anchor_utils import AnchorGenerator
from yolort.models.box_head import YOLOHead


__all__ = ['yolov5lite']


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

        out_channels = [64, 128, 256]

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

        # compute the yolo heads outputs using the features
        head_outputs = self.head(features)

        return head_outputs


def _extra_block(in_channels: int, out_channels: int, norm_layer: Callable[..., nn.Module]) -> nn.Sequential:
    activation = nn.ReLU6
    intermediate_channels = out_channels // 2
    return nn.Sequential(
        # 1x1 projection to half output channels
        ConvNormActivation(in_channels, intermediate_channels, kernel_size=1,
                           norm_layer=norm_layer, activation_layer=activation),

        # 3x3 depthwise with stride 2 and padding 1
        ConvNormActivation(intermediate_channels, intermediate_channels, kernel_size=3, stride=2,
                           groups=intermediate_channels, norm_layer=norm_layer, activation_layer=activation),

        # 1x1 projetion to output channels
        ConvNormActivation(intermediate_channels, out_channels, kernel_size=1,
                           norm_layer=norm_layer, activation_layer=activation),
    )


def _normal_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.03)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)


class YOLOv5LiteFeatureExtractorMobileNet(nn.Module):
    def __init__(self, backbone: nn.Module, c4_pos: int, norm_layer: Callable[..., nn.Module], width_mult: float = 1.0,
                 min_depth: int = 16, **kwargs: Any):
        super().__init__()

        assert not backbone[c4_pos].use_res_connect
        self.features = nn.Sequential(
            # As described in section 6.3 of MobileNetV3 paper
            nn.Sequential(*backbone[:c4_pos], backbone[c4_pos].block[0]),  # from start until C4 expansion layer
            nn.Sequential(backbone[c4_pos].block[1:], *backbone[c4_pos + 1:]),  # from C4 depthwise until end
        )

        get_depth = lambda d: max(min_depth, int(d * width_mult))  # noqa: E731
        extra = nn.ModuleList([
            _extra_block(backbone[-1].out_channels, get_depth(512), norm_layer),
            _extra_block(get_depth(512), get_depth(256), norm_layer),
            _extra_block(get_depth(256), get_depth(256), norm_layer),
            _extra_block(get_depth(256), get_depth(128), norm_layer),
        ])
        _normal_init(extra)

        self.extra = extra

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # Get feature maps from backbone and extra. Can't be refactored due to JIT limitations.
        output = []
        for block in self.features:
            x = block(x)
            output.append(x)

        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])


def _mobilenet_extractor(backbone_name: str, progress: bool, pretrained: bool, trainable_layers: int,
                         norm_layer: Callable[..., nn.Module], **kwargs: Any):
    backbone = mobilenet.__dict__[backbone_name](pretrained=pretrained, progress=progress,
                                                 norm_layer=norm_layer, **kwargs).features
    if not pretrained:
        # Change the default initialization scheme if not pretrained
        _normal_init(backbone)

    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    num_stages = len(stage_indices)

    # find the index of the layer from which we wont freeze
    assert 0 <= trainable_layers <= num_stages
    freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    return YOLOv5LiteFeatureExtractorMobileNet(backbone, stage_indices[-2], norm_layer, **kwargs)


def yolov5lite(
    pretrained: bool = False,
    progress: bool = True,
    num_classes: int = 91,
    pretrained_backbone: bool = False,
    trainable_backbone_layers: Optional[int] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    **kwargs: Any,
):
    """Constructs an YOLOv5lite model and a MobileNetV3 Large backbone, as described at
    `"Searching for MobileNetV3"
    <https://arxiv.org/abs/1905.02244>`_ and
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks"
    <https://arxiv.org/abs/1801.04381>`_.

    See :func:`~torchvision.models.detection.ssd300_vgg16` for more details.

    Example:

        >>> model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 320, 320), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 6, with 6 meaning all backbone layers are trainable.
        norm_layer (callable, optional): Module specifying the normalization layer to use.
    """

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 6, 6
    )

    if pretrained:
        pretrained_backbone = False

    # Enable reduced tail if no pretrained backbone is selected. See Table 6 of MobileNetV3 paper.
    reduce_tail = not pretrained_backbone

    if norm_layer is None:
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    backbone = _mobilenet_extractor(
        "mobilenet_v3_large",
        progress,
        pretrained_backbone,
        trainable_backbone_layers,
        norm_layer,
        reduced_tail=reduce_tail,
        **kwargs,
    )

    model = YOLOv5Lite(backbone, num_classes)

    return model
