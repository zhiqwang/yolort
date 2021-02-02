# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import torch
from torch import nn, Tensor

from .common import Conv, BottleneckCSP

from typing import Callable, List, Dict, Optional


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
        depth_multiple: float,
        block: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        assert len(in_channels_list) == 3, "currently only support length 3."

        if block is None:
            block = BottleneckCSP

        depth_gain = max(round(3 * depth_multiple), 1)

        inner_blocks = [
            block(in_channels_list[2], in_channels_list[2], n=depth_gain, shortcut=False),
            Conv(in_channels_list[2], in_channels_list[1], 1, 1),
            nn.Upsample(scale_factor=2),
            block(in_channels_list[2], in_channels_list[1], n=depth_gain, shortcut=False),
            Conv(in_channels_list[1], in_channels_list[0], 1, 1),
            nn.Upsample(scale_factor=2),
        ]

        self.inner_blocks = nn.ModuleList(inner_blocks)

        layer_blocks = [
            block(in_channels_list[1], in_channels_list[0], n=depth_gain, shortcut=False),
            Conv(in_channels_list[0], in_channels_list[0], 3, 2),
            block(in_channels_list[1], in_channels_list[1], n=depth_gain, shortcut=False),
            Conv(in_channels_list[1], in_channels_list[1], 3, 2),
            block(in_channels_list[2], in_channels_list[2], n=depth_gain, shortcut=False),
        ]
        self.layer_blocks = nn.ModuleList(layer_blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
            elif isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6)):
                m.inplace = True

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x: Dict[str, Tensor]) -> List[Tensor]:
        """
        Computes the PAN for a set of feature maps.

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after PAN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        x = list(x.values())

        # Descending the feature pyramid
        inners = []
        last_inner = self.get_result_from_inner_blocks(x[2], 0)
        last_inner = self.get_result_from_inner_blocks(last_inner, 1)
        inners.append(last_inner)
        last_inner = self.get_result_from_inner_blocks(last_inner, 2)
        last_inner = torch.cat([last_inner, x[1]], dim=1)
        last_inner = self.get_result_from_inner_blocks(last_inner, 3)
        last_inner = self.get_result_from_inner_blocks(last_inner, 4)
        inners.insert(0, last_inner)
        last_inner = self.get_result_from_inner_blocks(last_inner, 5)
        last_inner = torch.cat([last_inner, x[0]], dim=1)
        inners.insert(0, last_inner)

        # Ascending the feature pyramid
        results = []
        last_inner = self.get_result_from_layer_blocks(inners[0], 0)
        results.append(last_inner)

        for idx in range(len(inners) - 1):
            last_inner = self.get_result_from_layer_blocks(last_inner, 2 * idx + 1)
            last_inner = torch.cat([last_inner, inners[idx + 1]], dim=1)
            last_inner = self.get_result_from_layer_blocks(last_inner, 2 * idx + 2)
            results.append(last_inner)

        return results
