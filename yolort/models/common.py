# Modified from ultralytics/yolov5 by Zhiqiang Wang
# This file contains modules common to various models
import math
from typing import List

import torch
from torch import nn, Tensor


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, version='r4.0'):
        """
        Args:
            c1 (int): ch_in
            c2 (int): ch_out
            k (int): kernel
            s (int): stride
            p (Optional[int]): padding
            g (int): groups
            act (bool): determine the activation function
            version (str): ultralytics release version: r3.1 or r4.0
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        if version == 'r4.0':
            self.act = nn.SiLU() if act else nn.Identity()
        elif version == 'r3.1':
            self.act = nn.Hardswish() if act else nn.Identity()
        else:
            raise NotImplementedError("Currently only supports version r3.1 and r4.0")

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, version='r4.0'):
        """
        Args:
            c1 (int): ch_in
            c2 (int): ch_out
            shortcut (bool): shortcut
            g (int): groups
            e (float): expansion
            version (str): ultralytics release version: r3.1 or r4.0
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, version=version)
        self.cv2 = Conv(c_, c2, 3, 1, g=g, version=version)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Args:
            c1 (int): ch_in
            c2 (int): ch_out
            n (int): number
            shortcut (bool): shortcut
            g (int): groups
            e (float): expansion
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, version='r3.1')
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1, version='r3.1')
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0, version='r3.1') for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Args:
            c1 (int): ch_in
            c2 (int): ch_out
            n (int): number
            shortcut (bool): shortcut
            g (int): groups
            e (float): expansion
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), version='r4.0'):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, version=version)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1, version=version)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher

    equivalent to SPP(k=(5, 9, 13))
    """
    def __init__(self, c1, c2, k=5, version='r4.0'):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, version=version)
        self.cv2 = Conv(c_ * 4, c2, 1, 1, version=version)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, version='r4.0'):
        """
        Args:
            c1 (int): ch_in
            c2 (int): ch_out
            k (int): kernel
            s (int): stride
            p (Optional[int]): padding
            g (int): groups
            act (bool): determine the activation function
            version (str): ultralytics release version: r3.1 or r4.0
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act, version=version)

    def forward(self, x: Tensor) -> Tensor:
        y = focus_transform(x)
        out = self.conv(y)

        return out


def focus_transform(x: Tensor) -> Tensor:
    '''x(b,c,w,h) -> y(b,4c,w/2,h/2)'''
    y = torch.cat([x[..., ::2, ::2],
                   x[..., 1::2, ::2],
                   x[..., ::2, 1::2],
                   x[..., 1::2, 1::2]], 1)
    return y


def space_to_depth(x: Tensor) -> Tensor:
    '''x(b,c,w,h) -> y(b,4c,w/2,h/2)'''
    N, C, H, W = x.size()
    x = x.reshape(N, C, H // 2, 2, W // 2, 2)
    x = x.permute(0, 5, 3, 1, 2, 4)
    y = x.reshape(N, C * 4, H // 2, W // 2)
    return y


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, x: List[Tensor]) -> Tensor:
        if isinstance(x, Tensor):
            prev_features = [x]
        else:
            prev_features = x
        return torch.cat(prev_features, self.d)


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """
        Args:
            c1 (int): ch_in
            c2 (int): ch_out
            k (int): kernel
            s (int): stride
            p (Optional[int]): padding
            g (int): groups
        """
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  # to x(b,c2,1,1)
        self.flat = Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)
