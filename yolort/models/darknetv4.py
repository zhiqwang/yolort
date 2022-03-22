# Copyright (c) 2021, yolort team. All rights reserved.

from typing import Any, List, Callable, Optional

import torch
from torch import nn, Tensor
from yolort.v5 import Conv, Focus, BottleneckCSP, C3, SPP

from ._api import WeightsEnum
from ._utils import _make_divisible, _ovewrite_named_param


__all__ = [
    "DarkNetV4",
    "darknet_s_r3_1",
    "darknet_m_r3_1",
    "darknet_l_r3_1",
    "darknet_s_r4_0",
    "darknet_m_r4_0",
    "darknet_l_r4_0",
]


class DarkNetV4(nn.Module):
    """
    DarkNetV4 main class

    Args:
        depth_multiple (float): Depth multiplier
        width_multiple (float): Width multiplier - adjusts number of channels
            in each layer by this amount
        version (str): Module version released by ultralytics, set to r4.0.
        block: Module specifying inverted residual building block for darknet
        stages_repeats (Optional[List[int]]): List of repeats number in the stages.
        stages_out_channels (Optional[List[int]]): List of channels number in the stages.
        num_classes (int): Number of classes
        round_nearest (int): Round the number of channels in each layer to be
            a multiple of this number. Set to 1 to turn off rounding
        last_channel (int): Number of the last channel
    """

    def __init__(
        self,
        depth_multiple: float,
        width_multiple: float,
        version: str = "r4.0",
        block: Optional[Callable[..., nn.Module]] = None,
        stages_repeats: Optional[List[int]] = None,
        stages_out_channels: Optional[List[int]] = None,
        num_classes: int = 1000,
        round_nearest: int = 8,
        last_channel: int = 1024,
    ) -> None:
        super().__init__()

        if version not in ["r3.1", "r4.0"]:
            raise NotImplementedError(
                f"Currently does not support version: {version}. Feel free to file an issue "
                "labeled enhancement to us."
            )

        if block is None:
            block = _block[version]

        input_channel = 64

        if stages_repeats is None:
            stages_repeats = [3, 9, 9]

        if stages_out_channels is None:
            stages_out_channels = [128, 256, 512]

        # Initial an empty features list
        layers: List[nn.Module] = []

        # building first layer
        out_channel = _make_divisible(input_channel * width_multiple, round_nearest)
        layers.append(Focus(3, out_channel, k=3, version=version))
        input_channel = out_channel

        # building CSP blocks
        for depth_gain, out_channel in zip(stages_repeats, stages_out_channels):
            depth_gain = max(round(depth_gain * depth_multiple), 1)
            out_channel = _make_divisible(out_channel * width_multiple, round_nearest)
            layers.append(Conv(input_channel, out_channel, k=3, s=2, version=version))
            layers.append(block(out_channel, out_channel, n=depth_gain))
            input_channel = out_channel

        # building last CSP blocks
        last_channel = _make_divisible(last_channel * width_multiple, round_nearest)
        layers.append(Conv(input_channel, last_channel, k=3, s=2, version=version))
        layers.append(SPP(last_channel, last_channel, k=(5, 9, 13), version=version))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
            elif isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6)):
                m.inplace = True

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


_block = {
    "r3.1": BottleneckCSP,
    "r4.0": C3,
}


def _darknet_v4_conf(weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> DarkNetV4:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = DarkNetV4(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def darknet_s_r3_1(
    *, weights: Optional[WeightsEnum] = None, progress: bool = True, **kwargs: Any
) -> DarkNetV4:
    """
    Constructs the DarkNet release 3.1 model with small channels.

    Args:
        weights (WeightsEnum, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _darknet_v4_conf(weights, progress, 0.33, 0.5, version="r3.1", **kwargs)


def darknet_m_r3_1(
    *, weights: Optional[WeightsEnum] = None, progress: bool = True, **kwargs: Any
) -> DarkNetV4:
    """
    Constructs the DarkNet release 3.1 model with medium channels.

    Args:
        weights (WeightsEnum, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _darknet_v4_conf(weights, progress, 0.67, 0.75, version="r3.1", **kwargs)


def darknet_l_r3_1(
    *, weights: Optional[WeightsEnum] = None, progress: bool = True, **kwargs: Any
) -> DarkNetV4:
    """
    Constructs the DarkNet release 3.1 model with large channels.

    Args:
        weights (WeightsEnum, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _darknet_v4_conf(weights, progress, 1.0, 1.0, version="r3.1", **kwargs)


def darknet_s_r4_0(
    *, weights: Optional[WeightsEnum] = None, progress: bool = True, **kwargs: Any
) -> DarkNetV4:
    """
    Constructs the DarkNet release 4.0 model with small channels.

    Args:
        weights (WeightsEnum, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _darknet_v4_conf(weights, progress, 0.33, 0.5, version="r4.0", **kwargs)


def darknet_m_r4_0(
    *, weights: Optional[WeightsEnum] = None, progress: bool = True, **kwargs: Any
) -> DarkNetV4:
    """
    Constructs the DarkNet release 4.0 model with medium channels.

    Args:
        weights (WeightsEnum, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _darknet_v4_conf(weights, progress, 0.67, 0.75, version="r4.0", **kwargs)


def darknet_l_r4_0(
    *, weights: Optional[WeightsEnum] = None, progress: bool = True, **kwargs: Any
) -> DarkNetV4:
    """
    Constructs the DarkNet release 4.0 model with large channels.

    Args:
        weights (WeightsEnum, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _darknet_v4_conf(weights, progress, 1.0, 1.0, version="r4.0", **kwargs)
