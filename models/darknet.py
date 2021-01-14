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
        channels_list_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """
        MobileNet V2 main class

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

        input_channel = 32
        last_channel = 1024

        # only check the first element, assuming user knows t,c,n,s are required
        if len(channels_list_setting) == 0 or len(channels_list_setting[0]) != 4:
            raise ValueError("channels_list_setting should be non-empty "
                             "or a 4-element list, got {}".format(channels_list_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_multiple, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_multiple), round_nearest)
        layers: List[nn.Module] = [Focus(3, input_channel, k=3)]

        # building CSP blocks
        for i, cnf in enumerate(channels_list_setting):
            layers.append(Conv)
            if i == 3:
                layers.append(SPP)

            if i in [4, 5]:
                layers.append(nn.Upsample)

            if i in [4, 5, 6, 7]:
                layers.append(Concat)  # Cat backbone, set as the FPN modules

            layers.append(block(cnf))

        # make it nn.Sequential
        self.features = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x


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


