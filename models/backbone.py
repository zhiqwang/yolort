# Modified from ultralytics/yolov5 by Zhiqiang Wang
from collections import OrderedDict

import yaml

import torch
from torch import nn, Tensor
from torch.jit.annotations import List, Dict

from .common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, Concat
from .experimental import MixConv2d, CrossConv, C3
from .box_head import YoloHead as Detect

from utils.general import make_divisible
from utils.torch_utils import fuse_conv_and_bn, model_info, initialize_weights


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
            save_list=yolo_body.save,
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor):
        x = self.body(x)
        out: List[Tensor] = []

        for name, feature in x.items():
            out.append(feature)

        return out


class YoloBody(nn.Module):
    def __init__(self, model_dict, channels=3, num_classes=None):
        super().__init__()
        # Define model
        self.model, self.save = parse_model(model_dict, channels=[channels])

        # Init weights, biases
        initialize_weights(self)
        self.info()

    def forward(self, x: Tensor) -> Tensor:
        out = x
        y = torch.jit.annotate(List[Tensor], [])

        for i, m in enumerate(self.model):
            if m.f > 0:  # Concat layer
                out = torch.cat([out, y[sorted(self.save).index(m.f)]], 1)
            else:
                out = m(out)  # run
            if i in self.save:
                y.append(out)  # save output
        return out

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def info(self, verbose=False):  # print model information
        model_info(self, verbose)


def parse_model(model_dict, channels):
    anchors, num_classes = model_dict['anchors'], model_dict['nc']
    num_anchors = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    num_outputs = num_anchors * (num_classes + 5)

    layers, save, c2 = [], [], channels[-1]  # layers, savelist, channels out
    for i, (f, n, m, args) in enumerate(model_dict['backbone'] + model_dict['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = max(round(n * model_dict['depth_multiple']), 1) if n > 1 else n  # depth gain
        if m in [Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = channels[f], args[0]
            c2 = make_divisible(c2 * model_dict['width_multiple'], 8) if c2 != num_outputs else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [channels[f]]
        elif m is Concat:
            c2 = sum([channels[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            continue
        else:
            c2 = channels[f]

        module = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        module.f = -1 if f == -1 else f[-1]

        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(module)
        channels.append(c2)
    return nn.Sequential(*layers), save


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
        model_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    backbone = YoloBody(model_dict)
    body = YoloBackbone(
        yolo_body=backbone,
        return_layers={'17': '0', '20': '1', '23': '2'},
        out_channels=[128, 256, 512],
    )
    return body
