# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from typing import List, Optional

from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from . import darknet
from .path_aggregation_network import PathAggregationNetwork


class BackboneWithPAN(nn.Module):
    """
    Adds a PAN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediateLayerGetter apply here.

    Args:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        depth_multiple (float): depth multiplier
        version (str): Module version released by ultralytics: ["r3.1", "r4.0", "r6.0"].
        use_p6 (bool): Whether to use P6 layers.

    Attributes:
        out_channels (int): the number of channels in the PAN
    """

    def __init__(
        self,
        backbone,
        return_layers,
        in_channels_list,
        depth_multiple,
        version,
        use_p6=False,
    ):
        super().__init__()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.pan = PathAggregationNetwork(
            in_channels_list,
            depth_multiple,
            version=version,
            use_p6=use_p6,
        )
        self.out_channels = in_channels_list

    def forward(self, x):
        x = self.body(x)
        x = self.pan(x)
        return x


def darknet_pan_backbone(
    backbone_name: str,
    depth_multiple: float,
    width_multiple: float,
    pretrained: Optional[bool] = False,
    returned_layers: Optional[List[int]] = None,
    version: str = "r6.0",
    use_p6: bool = False,
):
    """
    Constructs a specified DarkNet backbone with PAN on top. Freezes the specified number of
    layers in the backbone.

    Examples:

        >>> from models.backbone_utils import darknet_pan_backbone
        >>> backbone = darknet_pan_backbone("darknet_s_r4_0")
        >>> # get some dummy image
        >>> x = torch.rand(1, 3, 64, 64)
        >>> # compute the output
        >>> output = backbone(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        [('0', torch.Size([1, 128, 8, 8])),
         ('1', torch.Size([1, 256, 4, 4])),
         ('2', torch.Size([1, 512, 2, 2]))]

    Args:
        backbone_name (string): darknet architecture. Possible values are "darknet_s_r3_1",
            "darknet_m_r3_1", "darknet_l_r3_1", "darknet_s_r4_0", "darknet_m_r4_0",
            "darknet_l_r4_0", "darknet_s_r6_0", "darknet_m_r6_0", and "darknet_l_r6_0".
        pretrained (bool): If True, returns a model with backbone pre-trained on Imagenet
        version (str): Module version released by ultralytics. Possible values
            are ["r3.1", "r4.0", "r6.0"]. Default: "r6.0".
        use_p6 (bool): Whether to use P6 layers.
    """
    assert version in [
        "r3.1",
        "r4.0",
        "r6.0",
    ], "Currently only supports version 'r3.1', 'r4.0' and 'r6.0'."

    last_channel = 768 if use_p6 else 1024
    backbone = darknet.__dict__[backbone_name](
        pretrained=pretrained,
        last_channel=last_channel,
    ).features

    if returned_layers is None:
        returned_layers = [4, 6, 8]

    return_layers = {str(k): str(i) for i, k in enumerate(returned_layers)}

    grow_widths = [256, 512, 768, 1024] if use_p6 else [256, 512, 1024]
    in_channels_list = [int(gw * width_multiple) for gw in grow_widths]

    return BackboneWithPAN(
        backbone,
        return_layers,
        in_channels_list,
        depth_multiple,
        version,
        use_p6=use_p6,
    )
