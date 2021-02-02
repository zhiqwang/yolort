# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import torch
from torch import nn, Tensor
from torch.hub import load_state_dict_from_url

from .common import Conv, SPP, Focus, BottleneckCSP
from .experimental import C3

from typing import Callable, List, Optional, Any


__all__ = ['DarkNet', 'darknet_s_r3_1', 'darknet_m_r3_1', 'darknet_l_r3_1',
           'darknet_s_r4_0', 'darknet_m_r4_0', 'darknet_l_r4_0']

model_urls = {
    "darknet_s_r3.1": None,
    "darknet_m_r3.1": None,
    "darknet_l_r3.1": None,
    "darknet_s_r4.0": None,
    "darknet_m_r4.0": None,
    "darknet_l_r4.0": None,
}  # TODO: add checkpoint weights


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DarkNet(nn.Module):
    def __init__(
        self,
        depth_multiple: float,
        width_multiple: float,
        block: Optional[Callable[..., nn.Module]] = None,
        stages_repeats: Optional[List[int]] = None,
        stages_out_channels: Optional[List[int]] = None,
        num_classes: int = 1000,
        round_nearest: int = 8,
    ) -> None:
        """
        DarkNet main class

        Args:
            num_classes (int): Number of classes
            depth_multiple (float): Depth multiplier
            width_multiple (float): Width multiplier - adjusts number of channels in each layer by this amount
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for darknet
        """
        super().__init__()

        if block is None:
            block = BottleneckCSP

        input_channel = 64
        last_channel = 1024

        if stages_repeats is None:
            stages_repeats = [3, 9, 9]

        if stages_out_channels is None:
            stages_out_channels = [128, 256, 512]

        # Initial an empty features list
        layers: List[nn.Module] = []

        # building first layer
        out_channel = _make_divisible(input_channel * width_multiple, round_nearest)
        layers.append(Focus(3, out_channel, k=3))
        input_channel = out_channel

        # building CSP blocks
        for depth_gain, out_channel in zip(stages_repeats, stages_out_channels):
            depth_gain = max(round(depth_gain * depth_multiple), 1)
            out_channel = _make_divisible(out_channel * width_multiple, round_nearest)
            layers.append(Conv(input_channel, out_channel, k=3, s=2))
            layers.append(block(out_channel, out_channel, n=depth_gain))
            input_channel = out_channel

        # building last CSP blocks
        last_channel = _make_divisible(last_channel * width_multiple, round_nearest)
        layers.append(Conv(input_channel, last_channel, k=3, s=2))
        layers.append(SPP(last_channel, last_channel, k=(5, 9, 13)))

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


def _darknet(arch: str, pretrained: bool, progress: bool, *args: Any, **kwargs: Any) -> DarkNet:
    """
    Constructs a DarkNet architecture from
    # TODO

    """
    model = DarkNet(*args, **kwargs)

    if pretrained:
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)

    return model


_block = {
    "r3.1": BottleneckCSP,
    "r4.0": C3,
}


def darknet_s_r3_1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DarkNet:
    """
    Constructs a DarkNet with small channels, as described in release 3.1
    # TODO

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _darknet("darknet_s_r3.1", pretrained, progress,
                    0.33, 0.5, block=_block["r3.1"], **kwargs)


def darknet_m_r3_1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DarkNet:
    """
    Constructs a DarkNet with small channels, as described in release 3.1
    # TODO

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _darknet("darknet_m_r3.1", pretrained, progress,
                    0.67, 0.75, block=_block["r3.1"], **kwargs)


def darknet_l_r3_1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DarkNet:
    """
    Constructs a DarkNet with small channels, as described in release 3.1
    # TODO

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _darknet("darknet_l_r3.1", pretrained, progress,
                    1.0, 1.0, block=_block["r3.1"], **kwargs)


def darknet_s_r4_0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DarkNet:
    """
    Constructs a DarkNet with small channels, as described in release 3.1
    # TODO

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _darknet("darknet_s_r4.0", pretrained, progress,
                    0.33, 0.5, block=_block["r4.0"], **kwargs)


def darknet_m_r4_0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DarkNet:
    """
    Constructs a DarkNet with small channels, as described in release 3.1
    # TODO

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _darknet("darknet_m_r4.0", pretrained, progress,
                    0.67, 0.75, block=_block["r4.0"], **kwargs)


def darknet_l_r4_0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DarkNet:
    """
    Constructs a DarkNet with small channels, as described in release 3.1
    # TODO

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _darknet("darknet_l_r4.0", pretrained, progress,
                    1.0, 1.0, block=_block["r4.0"], **kwargs)
