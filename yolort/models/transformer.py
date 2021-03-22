# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
"""
The transformer attention network blocks.

Mostly copy-paste from <https://github.com/dingyiwei/yolov5/tree/Transformer>.
"""
from torch import nn

from .common import Conv, C3
from .path_aggregation_network import PathAggregationNetwork
from .backbone_utils import BackboneWithPAN

from . import darknet

from typing import Callable, List, Optional


def darknet_tan_backbone(
    backbone_name: str,
    depth_multiple: float,
    width_multiple: float,
    pretrained: Optional[bool] = False,
    returned_layers: Optional[List[int]] = None,
    version: str = 'r4.0',
):
    """
    Constructs a specified DarkNet backbone with TAN on top. Freezes the specified number of
    layers in the backbone.

    Examples::

        >>> from models.backbone_utils import darknet_tan_backbone
        >>> backbone = darknet_tan_backbone('darknet3_1', pretrained=True, trainable_layers=3)
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
        backbone_name (string): darknet architecture. Possible values are 'DarkNet', 'darknet_s_r3_1',
           'darknet_m_r3_1', 'darknet_l_r3_1', 'darknet_s_r4_0', 'darknet_m_r4_0', 'darknet_l_r4_0'
        norm_layer (torchvision.ops): it is recommended to use the default value. For details visit:
            (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        pretrained (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_layers (int): number of trainable (not frozen) darknet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
        version (str): ultralytics release version, currently only supports r3.1 or r4.0
    """
    backbone = darknet.__dict__[backbone_name](pretrained=pretrained).features

    if returned_layers is None:
        returned_layers = [4, 6, 8]

    return_layers = {str(k): str(i) for i, k in enumerate(returned_layers)}

    in_channels_list = [int(gw * width_multiple) for gw in [256, 512, 1024]]

    return BackboneWithTAN(backbone, return_layers, in_channels_list, depth_multiple, version)


class BackboneWithTAN(BackboneWithPAN):
    """
    Adds a TAN on top of a model.
    """
    def __init__(self, backbone, return_layers, in_channels_list, depth_multiple, version):
        super().__init__(backbone, return_layers, in_channels_list, depth_multiple, version)
        self.pan = TransformerAttentionNetwork(
            in_channels_list,
            depth_multiple,
            version=version,
        )


class TransformerAttentionNetwork(PathAggregationNetwork):
    def __init__(
        self,
        in_channels_list: List[int],
        depth_multiple: float,
        version: str,
        block: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__(in_channels_list, depth_multiple, version=version, block=block)
        assert len(in_channels_list) == 3, "currently only support length 3."

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


class C3TR(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class TransformerLayer(nn.Module):
    def __init__(self, c, num_heads):
        """
        Args:
            c (int): number of channels
            num_heads: number of heads
        """
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)

        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers):
        """
        Args:
            c1 (int): number of input channels
            c2 (int): number of output channels
            num_heads: number of heads
            num_layers: number of layers
        """
        super().__init__()

        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)

        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)

        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x
