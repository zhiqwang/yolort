# Modified from ultralytics/yolov5 by Zhiqiang Wang
from collections import OrderedDict

import yaml

import torch
from torch import nn, Tensor
from torch.jit.annotations import List, Dict, Optional

from .common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, Concat
from .experimental import MixConv2d, CrossConv, C3
from .box_head import YoloHead as Detect


class YoloBackbone(nn.Module):
    def __init__(
        self,
        yolo_body: nn.Module,
        return_layers: dict,
        out_channels: List[int],
    ):
        super().__init__()
        self.body = IntermediateLayerGetter(
            yolo_body.model,
            return_layers=return_layers,
            save_list=yolo_body.save_list,
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor):
        x = self.body(x)
        out: List[Tensor] = []

        for name, feature in x.items():
            out.append(feature)

        return out


class YoloBody(nn.Module):
    __annotations__ = {
        "save_list": List[int],
    }

    def __init__(self, layers, save_list):
        super().__init__()
        # Define model
        self.model = nn.Sequential(*layers)
        self.save_list = save_list

        # Init weights, biases
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        out = x
        y = torch.jit.annotate(List[Tensor], [])

        for i, m in enumerate(self.model):
            if m.f > 0:  # Concat layer
                out = torch.cat([out, y[sorted(self.save_list).index(m.f)]], 1)
            else:
                out = m(out)  # run
            if i in self.save_list:
                y.append(out)  # save output
        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
            elif isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6)):
                m.inplace = True


def parse_model(model_dict, in_channels=3):
    head_info = ()
    anchors, num_classes = model_dict['anchors'], model_dict['nc']
    num_anchors = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    num_outputs = num_anchors * (num_classes + 5)

    c2 = in_channels
    layers, save_list, channels = [], [], [c2]  # layers, save list, channels out
    # from, number, module, args
    for i, (f, n, m, args) in enumerate(model_dict['backbone'] + model_dict['head']):
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = max(round(n * model_dict['depth_multiple']), 1) if n > 1 else n  # depth gain
        if m in [Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = channels[f], args[0]
            c2 = _make_divisible(c2 * model_dict['width_multiple'], 8) if c2 != num_outputs else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [channels[f]]
        elif m is Concat:
            c2 = sum([channels[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            num_layers, anchor_grids = f, args[-1]
            out_channels = [channels[x + 1] for x in f]
            head_info = (out_channels, anchor_grids, num_layers)
            continue
        else:
            c2 = channels[f]

        module = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        module.f = -1 if f == -1 else f[-1]

        save_list.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(module)
        channels.append(c2)
    return layers, save_list, head_info


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


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
        "save_list": List[int],
    }

    def __init__(self, model, return_layers, save_list):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers
        self.save_list = save_list

    def forward(self, x):
        out = OrderedDict()
        y = torch.jit.annotate(List[Tensor], [])

        for i, (name, module) in enumerate(self.items()):
            if module.f > 0:  # Concat layer
                x = torch.cat([x, y[sorted(self.save_list).index(module.f)]], 1)
            else:
                x = module(x)  # run
            if i in self.save_list:
                y.append(x)  # save output

            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


def darknet(cfg_path='./models/yolov5s.yaml', pretrained=False):
    with open(cfg_path) as f:
        model_dict = yaml.load(f, Loader=yaml.FullLoader)

    layers, save_list, head_info = parse_model(model_dict, in_channels=3)

    backbone = YoloBody(layers, save_list)

    body = YoloBackbone(
        yolo_body=backbone,
        return_layers={str(key): str(i) for i, key in enumerate(head_info[2])},
        out_channels=head_info[0],
    )
    return body, head_info[1]
