# Copyright (c) 2021, yolort team. All rights reserved.

from typing import Dict, List, Callable, Optional

import torch
from torch import nn, Tensor
from yolort.v5 import Conv, BottleneckCSP, C3, SPP


class IntermediateLevelP6(nn.Module):
    """
    This module is used to generate intermediate P6 block to the PAN.

    Args:
        x (List[Tensor]): the original feature maps

    Returns:
        results (List[Tensor]): the extended set of results
            of the PAN
    """

    def __init__(self, depth_multiple: float, in_channel: int, out_channel: int, version: str = "r4.0"):
        super().__init__()

        block = _block[version]
        depth_gain = max(round(3 * depth_multiple), 1)

        self.p6 = nn.Sequential(
            Conv(in_channel, out_channel, k=3, s=2, version=version),
            block(out_channel, out_channel, n=depth_gain),
        )

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        x.append(self.p6(x[-1]))
        return x


class PathAggregationNetwork(nn.Module):
    """
    Module that adds a PAN from on top of a set of feature maps. This is based on
    `"Path Aggregation Network for Instance Segmentation" <https://arxiv.org/abs/1803.01534>`_.

    The feature maps are currently supposed to be in increasing depth
    order.

    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the PAN will be added.

    Args:
        in_channels (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the PAN representation
        version (str): ultralytics release version: ["r3.1", "r4.0", "r6.0"]

    Examples:

        >>> m = PathAggregationNetwork()
        >>> # get some dummy data
        >>> x = OrderedDict()
        >>> x['feat0'] = torch.rand(1, 128, 52, 44)
        >>> x['feat2'] = torch.rand(1, 256, 26, 22)
        >>> x['feat3'] = torch.rand(1, 512, 13, 11)
        >>> # compute the PAN on top of x
        >>> output = m(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        [('feat0', torch.Size([1, 128, 52, 44])),
         ('feat2', torch.Size([1, 256, 26, 22])),
         ('feat3', torch.Size([1, 512, 13, 11]))]
    """

    def __init__(
        self,
        in_channels: List[int],
        depth_multiple: float,
        version: str = "r4.0",
        block: Optional[Callable[..., nn.Module]] = None,
        use_p6: bool = False,
    ):
        super().__init__()

        module_version = "r4.0" if version == "r6.0" else version

        # Define the Intermediate Block if necessary
        if use_p6:
            assert len(in_channels) == 4, "Length of in channels should be 4."
            intermediate_blocks = IntermediateLevelP6(
                depth_multiple, in_channels[2], in_channels[3], version=module_version
            )
        else:
            assert len(in_channels) == 3, "Length of in channels should be 3."
            intermediate_blocks = None

        self.intermediate_blocks = intermediate_blocks

        if block is None:
            block = _block[module_version]

        depth_gain = max(round(3 * depth_multiple), 1)

        if version == "r6.0":
            init_block = SPP(in_channels[-1], in_channels[-1], k=(5, 9, 13))
        elif version in ["r3.1", "r4.0"]:
            init_block = block(in_channels[-1], in_channels[-1], n=depth_gain, shortcut=False)
        else:
            raise NotImplementedError(f"Version {version} is not implemented yet.")

        # Define the inner blocks
        inner_blocks = [init_block]

        if use_p6:
            in_channel = in_channels[1] + in_channels[-1]
            inner_blocks_p6 = [
                Conv(in_channels[-1], in_channels[2], 1, 1, version=module_version),
                nn.Upsample(scale_factor=2),
                block(in_channel, in_channels[2], n=depth_gain, shortcut=False),
            ]
            inner_blocks.extend(inner_blocks_p6)

        inner_blocks.extend(
            [
                Conv(in_channels[2], in_channels[1], 1, 1, version=module_version),
                nn.Upsample(scale_factor=2),
                block(in_channels[-1], in_channels[1], n=depth_gain, shortcut=False),
                Conv(in_channels[1], in_channels[0], 1, 1, version=module_version),
                nn.Upsample(scale_factor=2),
            ]
        )
        self.inner_blocks = nn.ModuleList(inner_blocks)

        # Define the layer blocks
        layer_blocks = [
            block(in_channels[1], in_channels[0], n=depth_gain, shortcut=False),
            Conv(in_channels[0], in_channels[0], 3, 2, version=module_version),
            block(in_channels[1], in_channels[1], n=depth_gain, shortcut=False),
            Conv(in_channels[1], in_channels[1], 3, 2, version=module_version),
            block(in_channels[-1], in_channels[2], n=depth_gain, shortcut=False),
        ]

        if use_p6:
            in_channel = in_channels[1] + in_channels[-1]
            layer_blocks_p6 = [
                Conv(in_channels[2], in_channels[2], 3, 2, version=module_version),
                block(in_channel, in_channels[-1], n=depth_gain, shortcut=False),
            ]
            layer_blocks.extend(layer_blocks_p6)

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
        if self.intermediate_blocks is not None:
            x = self.intermediate_blocks(x)

        # Descending the feature pyramid
        num_features = len(x)
        inners = []
        last_inner = x[-1]
        for idx in range(num_features - 1):
            last_inner = self.get_result_from_inner_blocks(last_inner, 3 * idx)
            last_inner = self.get_result_from_inner_blocks(last_inner, 3 * idx + 1)
            inners.insert(0, last_inner)
            last_inner = self.get_result_from_inner_blocks(last_inner, 3 * idx + 2)
            last_inner = torch.cat([last_inner, x[num_features - idx - 2]], dim=1)

        inners.insert(0, last_inner)

        # Ascending the feature pyramid
        results = []
        last_inner = self.get_result_from_layer_blocks(inners[0], 0)
        results.append(last_inner)

        for idx in range(num_features - 1):
            last_inner = self.get_result_from_layer_blocks(last_inner, 2 * idx + 1)
            last_inner = torch.cat([last_inner, inners[idx + 1]], dim=1)
            last_inner = self.get_result_from_layer_blocks(last_inner, 2 * idx + 2)
            results.append(last_inner)

        return results


_block = {
    "r3.1": BottleneckCSP,
    "r4.0": C3,
}
