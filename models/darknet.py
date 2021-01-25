import torch
from torch import dropout, nn, Tensor
from torch.hub import load_state_dict_from_url

from .common import Concat, Conv, SPP, Focus, BottleneckCSP
from .experimental import C3

from typing import Callable, List, Optional, Any

__all__ = ['DarkNet', 'darknet3_1', 'darknet4_0']

_MODEL_URLS = {
    "3.1": None,
    "4.0": None,
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
        depth_multiple: float = 0.33,
        width_multiple: float = 0.5,
        channels_list_setting: Optional[List[List[List[int]]]] = None,
        num_classes: int = 1000,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """
        DarkNet main class

        Args:
            num_classes (int): Number of classes
            depth_multiple (float): Depth multiplier
            width_multiple (float): Width multiplier - adjusts number of channels in each layer by this amount
            channels_list_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for darknet
        """
        super().__init__()

        if block is None:
            block = BottleneckCSP

        input_channel = 64
        last_channel = 1024

        if channels_list_setting is None:
            channels_list_setting = [
                [[32, 64, 3, 2], [64, 64, 1]],  # P2/4
                [[64, 128, 3, 2], [128, 128, 3]],  # P3/8
                [[128, 256, 3, 2], [256, 256, 3]],  # P4/16
            ]

        # Initial an empty features list
        layers: List[nn.Module] = []

        # building first layer
        input_channel = _make_divisible(input_channel * width_multiple, round_nearest)
        layers.append(Focus(3, input_channel, k=3))

        # building CSP blocks
        for cfgs in channels_list_setting:
            layers.append(Conv(*cfgs[0]))
            layers.append(BottleneckCSP(*cfgs[1]))

        # building last CSP blocks
        last_channel = _make_divisible(last_channel * width_multiple, round_nearest)
        layers.append(Conv(256, last_channel, k=3, s=2))
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _darknet(
    arch: str,
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> DarkNet:
    """
    Constructs a DarkNet architecture from
    `"DarkNet: `_. # TODO

    """
    _BLOCK = {
        "3.1": BottleneckCSP,
        "4.0": C3,
    }

    model = DarkNet(block=_BLOCK[arch], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(_MODEL_URLS[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def darknet3_1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DarkNet:
    return _darknet("3.1", pretrained, progress, **kwargs)


def darknet4_0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DarkNet:
    return _darknet("4.0", pretrained, progress, **kwargs)
