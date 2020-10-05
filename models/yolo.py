import logging

from copy import deepcopy
from pathlib import Path
from typing import List

import torch
from torch import nn, Tensor

from models.common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, Concat
from models.experimental import MixConv2d, CrossConv, C3
from utils.general import make_divisible
from utils.torch_utils import fuse_conv_and_bn, model_info, initialize_weights

logger = logging.getLogger(__name__)


class Detect(nn.Module):
    def __init__(self, nc=80, stride=[8., 16., 32.], anchors=(), ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.stride = stride

    def get_result_from_m(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.m[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = 0
        for m in self.m:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.m:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x: List[Tensor]):
        # x = x.copy()  # for profiling
        device = x[0].device
        z: List[Tensor] = []  # inference output
        for i in range(self.nl):
            x[i] = self.get_result_from_m(x[i], i)  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if not isinstance(self.stride, Tensor):
                    self.stride = torch.tensor(self.stride, device=device)

                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx: int = 20, ny: int = 20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
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
            if type(m) is Conv and hasattr(Conv, 'bn'):
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
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
