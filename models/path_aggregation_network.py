from collections import OrderedDict
import torch

import torch.nn.functional as F
from torch import nn, Tensor, select

from .common import Conv, BottleneckCSP

from typing import Tuple, List, Dict, Optional


class PathAggregationNetwork(nn.Module):
    """
    Module that adds a PAN from on top of a set of feature maps. This is based on
    `"Path Aggregation Network for Instance Segmentation" <https://arxiv.org/abs/1803.01534>`_.

    The feature maps are currently supposed to be in increasing depth
    order.

    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the PAN will be added.

    Args:
        in_channels_list (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the PAN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names

    Examples::

        >>> m = PathAggregationNetwork()
        >>> # get some dummy data
        >>> x = OrderedDict()
        >>> x['feat0'] = torch.rand(1, 128, 52, 44)
        >>> x['feat2'] = torch.rand(1, 256, 26, 22)
        >>> x['feat3'] = torch.rand(1, 512, 13, 11)
        >>> # compute the PAN on top of x
        >>> output = m(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('feat0', torch.Size([1, 128, 52, 44])),
        >>>    ('feat2', torch.Size([1, 256, 26, 22])),
        >>>    ('feat3', torch.Size([1, 512, 13, 11]))]

    """
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
    ):
        super().__init__()
        assert len(in_channels_list) == 3, "current only support length 3."

        inner_blocks = [
            BottleneckCSP(in_channels_list[2], in_channels_list[2], n=1, shortcut=False),
            Conv(in_channels_list[2], in_channels_list[1], 1, 1),
            nn.Upsample(scale_factor=2),
            BottleneckCSP(in_channels_list[2], in_channels_list[1], n=1, shortcut=False),
            Conv(in_channels_list[1], in_channels_list[0], 1, 1),
            nn.Upsample(scale_factor=2),
        ]

        self.inner_blocks = nn.ModuleList(inner_blocks)

        layer_blocks = [
            BottleneckCSP(in_channels_list[1], in_channels_list[0], n=1, shortcut=False),
            Conv(in_channels_list[0], in_channels_list[0], 3, 2),
            BottleneckCSP(in_channels_list[1], in_channels_list[1], n=1, shortcut=False),
            Conv(in_channels_list[1], in_channels_list[1], 3, 2),
            BottleneckCSP(in_channels_list[2], in_channels_list[2], n=1, shortcut=False),
        ]
        self.layer_blocks = nn.ModuleList(layer_blocks)

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the PAN for a set of feature maps.

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after PAN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        inners = []

        last_inner = self.inner_blocks[0](x[2])
        last_inner = self.inner_blocks[1](last_inner)
        inners.append(last_inner)
        last_inner = self.inner_blocks[2](last_inner)
        last_inner = torch.cat([last_inner, x[1]], dim=1)
        last_inner = self.inner_blocks[3](last_inner)
        last_inner = self.inner_blocks[4](last_inner)
        inners.insert(0, last_inner)
        last_inner = self.inner_blocks[5](last_inner)
        last_inner = torch.cat([last_inner, x[0]], dim=1)
        inners.insert(0, last_inner)

        results = []
        last_inner = self.layer_blocks[0](inners[0])
        results.append(last_inner)
        last_inner = self.layer_blocks[1](last_inner)
        last_inner = torch.cat([last_inner, inners[1]], dim=1)
        last_inner = self.layer_blocks[2](last_inner)
        results.append(last_inner)
        last_inner = self.layer_blocks[3](last_inner)
        last_inner = torch.cat([last_inner, inners[2]], dim=1)
        last_inner = self.layer_blocks[4](last_inner)
        results.append(last_inner)

        return results
