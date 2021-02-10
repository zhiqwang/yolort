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
    The same limitations of IntermediatLayerGetter apply here.
    Args:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
    Attributes:
        out_channels (int): the number of channels in the PAN
    """
    def __init__(self, backbone, return_layers, in_channels_list, depth_multiple):
        super().__init__()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.pan = PathAggregationNetwork(in_channels_list, depth_multiple)
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
):
    """
    Constructs a specified ResNet backbone with PAN on top. Freezes the specified number of layers in the backbone.

    Examples::

        >>> from models.backbone_utils import darknet_pan_backbone
        >>> backbone = darknet_pan_backbone('darknet3_1', pretrained=True, trainable_layers=3)
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
        backbone_name (string): resnet architecture. Possible values are 'ResNet', 'resnet18', 'resnet34', 'resnet50',
             'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
        norm_layer (torchvision.ops): it is recommended to use the default value. For details visit:
            (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        pretrained (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    backbone = darknet.__dict__[backbone_name](pretrained=pretrained).features

    if returned_layers is None:
        returned_layers = [4, 6, 8]

    return_layers = {str(k): str(i) for i, k in enumerate(returned_layers)}

    in_channels_list = [int(gw * width_multiple) for gw in [256, 512, 1024]]

    return BackboneWithPAN(backbone, return_layers, in_channels_list, depth_multiple)
