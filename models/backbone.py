import logging
from collections import OrderedDict

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.jit.annotations import List, Dict

from .common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, Concat
from .experimental import MixConv2d, CrossConv, C3
from .box_head import YoloHead as Detect

from utils.general import make_divisible
from utils.torch_utils import fuse_conv_and_bn, model_info, initialize_weights

logger = logging.getLogger(__name__)


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
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nc and nc != self.yaml['nc']:
            print('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist, ch_out

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


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            continue
        else:
            c2 = ch[f]

        module = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in module.parameters()])  # number params
        module.f = -1 if f == -1 else f[-1]
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(module)
        ch.append(c2)
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
    backbone = YoloBody(cfg=cfg_path)
    body = YoloBackbone(
        yolo_body=backbone,
        return_layers={'17': '0', '20': '1', '23': '2'},
        out_channels=[128, 256, 512],
    )
    return body
