# Copyright (c) 2021, yolort team. All rights reserved.

from typing import Any, Callable, List, Optional

import torch
from torch import nn, Tensor
from yolort.utils import load_state_dict_from_url
from yolort.v5 import Conv, C3

from ._utils import _make_divisible


__all__ = [
    "DarkNetV6",
    "darknet_n_r6_0",
    "darknet_s_r6_0",
    "darknet_m_r6_0",
    "darknet_l_r6_0",
    "darknet_x_r6_0",
]

model_urls = {
    "darknet_n_r6.0": None,
    "darknet_s_r6.0": None,
    "darknet_m_r6.0": None,
    "darknet_l_r6.0": None,
    "darknet_x_r6.0": None,
}  # TODO: add checkpoint weights


class DarkNetV6(nn.Module):
    """
    DarkNetV6 main class.

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

        assert version == "r4.0", "Currently the module version used in DarkNetV6 is r4.0."

        if block is None:
            block = C3

        input_channel = 64

        if stages_repeats is None:
            stages_repeats = [3, 6, 9]

        if stages_out_channels is None:
            stages_out_channels = [128, 256, 512]

        # Initial an empty features list
        layers: List[nn.Module] = []

        # building first layer
        out_channel = _make_divisible(input_channel * width_multiple, round_nearest)
        layers.append(Conv(3, out_channel, k=6, s=2, p=2, version=version))
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
        depth_gain = max(round(3 * depth_multiple), 1)
        layers.append(block(last_channel, last_channel, n=depth_gain))

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


def _darknet_v6_conf(arch: str, pretrained: bool, progress: bool, *args: Any, **kwargs: Any) -> DarkNetV6:
    """
    Build a DarkNetV6 model.
    """
    model = DarkNetV6(*args, **kwargs)

    if pretrained:
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError(f"pretrained {arch} is not supported as of now")
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)

    return model


def darknet_n_r6_0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DarkNetV6:
    """
    Constructs the DarkNet release 6.0 model with nano channels.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _darknet_v6_conf("darknet_n_r6.0", pretrained, progress, 0.33, 0.25, **kwargs)


def darknet_s_r6_0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DarkNetV6:
    """
    Constructs the DarkNet release 6.0 model with small channels.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _darknet_v6_conf("darknet_s_r6.0", pretrained, progress, 0.33, 0.5, **kwargs)


def darknet_m_r6_0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DarkNetV6:
    """
    Constructs the DarkNet release 6.0 model with medium channels.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _darknet_v6_conf("darknet_m_r6.0", pretrained, progress, 0.67, 0.75, **kwargs)


def darknet_l_r6_0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DarkNetV6:
    """
    Constructs the DarkNet release 6.0 model with large channels.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _darknet_v6_conf("darknet_l_r6.0", pretrained, progress, 1.0, 1.0, **kwargs)


def darknet_x_r6_0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DarkNetV6:
    """
    Constructs the DarkNet release 6.0 model with X large channels.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _darknet_v6_conf("darknet_x_r6.0", pretrained, progress, 1.33, 1.25, **kwargs)
