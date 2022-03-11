# Copyright (c) 2021, yolort team. All rights reserved.

from typing import Callable, List, Optional

from torch import nn
from yolort.v5 import Conv, C3, C3TR

from . import darknet
from .backbone_utils import BackboneWithPAN
from .path_aggregation_network import PathAggregationNetwork


def darknet_tan_backbone(
    backbone_name: str,
    depth_multiple: float,
    width_multiple: float,
    pretrained: Optional[bool] = False,
    returned_layers: Optional[List[int]] = None,
    version: str = "r4.0",
    use_p6: bool = False,
):
    """
    Constructs a specified DarkNet backbone with TAN on top. Freezes the specified number of
    layers in the backbone.

    Examples:

        >>> from models.backbone_utils import darknet_tan_backbone
        >>> backbone = darknet_tan_backbone("darknet_s_r4_0")
        >>> # get some dummy image
        >>> x = torch.rand(1, 3, 64, 64)
        >>> # compute the output
        >>> output = backbone(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('0', torch.Size([1, 128, 8, 8])),
        >>>    ('1', torch.Size([1, 256, 4, 4])),
        >>>    ('2', torch.Size([1, 512, 2, 2]))]

    Args:
        backbone_name (string): darknet architecture. Possible values are "darknet_s_r4_0" Now.
        norm_layer (torchvision.ops): it is recommended to use the default value. For details visit:
            (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        pretrained (bool): If True, returns a model with backbone pre-trained on Imagenet
        version (str): Module version released by ultralytics, set to "r4.0".
    """
    assert version == "r4.0", "Currently only supports version r4.0."
    assert not use_p6, "Currently doesn't support the P6 structure."

    backbone = darknet.__dict__[backbone_name](pretrained=pretrained).features

    if returned_layers is None:
        returned_layers = [4, 6, 8]

    return_layers = {str(k): str(i) for i, k in enumerate(returned_layers)}

    in_channels_list = [int(gw * width_multiple) for gw in [256, 512, 1024]]

    return BackboneWithTAN(backbone, return_layers, in_channels_list, depth_multiple)


class BackboneWithTAN(BackboneWithPAN):
    """
    Adds a TAN on top of a model.
    """

    def __init__(self, backbone, return_layers, in_channels_list, depth_multiple):
        super().__init__(backbone, return_layers, in_channels_list, depth_multiple, "r4.0")
        self.pan = TransformerAttentionNetwork(
            in_channels_list,
            depth_multiple,
            version="r4.0",
        )


class TransformerAttentionNetwork(PathAggregationNetwork):
    def __init__(
        self,
        in_channels_list: List[int],
        depth_multiple: float,
        version: str = "r4.0",
        block: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__(in_channels_list, depth_multiple, version=version, block=block)
        assert len(in_channels_list) == 3, "Currently only supports length 3."
        assert version == "r4.0", "Currently only supports version r4.0."

        if block is None:
            block = C3

        depth_gain = max(round(3 * depth_multiple), 1)

        inner_blocks = [
            C3TR(in_channels_list[2], in_channels_list[2], n=depth_gain, shortcut=False),
            Conv(in_channels_list[2], in_channels_list[1], 1, 1, version=version),
            nn.Upsample(scale_factor=2),
            block(in_channels_list[2], in_channels_list[1], n=depth_gain, shortcut=False),
            Conv(in_channels_list[1], in_channels_list[0], 1, 1, version=version),
            nn.Upsample(scale_factor=2),
        ]

        self.inner_blocks = nn.ModuleList(inner_blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
            elif isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6)):
                m.inplace = True
