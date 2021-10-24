# YOLOv5 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import logging
import math
from copy import copy
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from torch import nn, Tensor
from torch.cuda import amp
from yolort.v5.utils.datasets import exif_transpose, letterbox
from yolort.v5.utils.general import (
    colorstr,
    increment_path,
    is_ascii,
    make_divisible,
    non_max_suppression,
    save_one_box,
    scale_coords,
    xyxy2xywh,
)
from yolort.v5.utils.plots import Annotator, colors
from yolort.v5.utils.torch_utils import time_sync

LOGGER = logging.getLogger(__name__)


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """
    Standard convolution

    Args:
        c1 (int): ch_in
        c2 (int): ch_out
        k (int): kernel
        s (int): stride
        p (Optional[int]): padding
        g (int): groups
        act (bool or nn.Module): determine the activation function
        version (str): Module version released by ultralytics. Possible values
            are ["r3.1", "r4.0"]. Default: "r4.0".
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, version="r4.0"):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        if version == "r4.0":
            self.act = nn.SiLU() if act else (act if isinstance(act, nn.Module) else nn.Identity())
        elif version == "r3.1":
            self.act = nn.Hardswish() if act else (act if isinstance(act, nn.Module) else nn.Identity())
        else:
            raise NotImplementedError(f"Currently doesn't support version {version}.")

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    """
    Depth-wise convolution class.

    Args:
        c1 (int): ch_in
        c2 (int): ch_out
        k (int): kernel
        s (int): stride
        act (bool or nn.Module): determine the activation function
        version (str): Module version released by ultralytics. Possible values
            are ["r3.1", "r4.0"]. Default: "r4.0".
    """

    def __init__(self, c1, c2, k=1, s=1, act=True, version="r4.0"):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act, version=version)


class Bottleneck(nn.Module):
    """
    Standard bottleneck

    Args:
        c1 (int): ch_in
        c2 (int): ch_out
        shortcut (bool): shortcut
        g (int): groups
        e (float): expansion
        version (str): Module version released by ultralytics. Possible values
            are ["r3.1", "r4.0"]. Default: "r4.0".
    """

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, version="r4.0"):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, version=version)
        self.cv2 = Conv(c_, c2, 3, 1, g=g, version=version)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """
    CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks

    Args:
        c1 (int): ch_in
        c2 (int): ch_out
        n (int): number
        shortcut (bool): shortcut
        g (int): groups
        e (float): expansion
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, version="r3.1")
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1, version="r3.1")
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0, version="r3.1") for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    """
    CSP Bottleneck with 3 convolutions

    Args:
        c1 (int): ch_in
        c2 (int): ch_out
        n (int): number
        shortcut (bool): shortcut
        g (int): groups
        e (float): expansion
        version (str): Module version released by ultralytics. Possible values
            are ["r4.0"]. Default: "r4.0".
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, version="r4.0"):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, version=version)
        self.cv2 = Conv(c1, c_, 1, 1, version=version)
        self.cv3 = Conv(2 * c_, c2, 1, version=version)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0, version=version) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), version="r4.0"):
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
    """

    def __init__(self, c1, c2, k=5, version="r4.0"):
        # Equivalent to SPP(k=(5, 9, 13)) when k=5
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
    """
    Focus wh information into c-space

    Args:
        c1 (int): ch_in
        c2 (int): ch_out
        k (int): kernel
        s (int): stride
        p (Optional[int]): padding
        g (int): groups
        act (bool or nn.Module): determine the activation function
        version (str): Module version released by ultralytics. Possible values
            are ["r3.1", "r4.0"]. Default: "r4.0".
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, version="r4.0"):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act, version=version)

    def forward(self, x: Tensor) -> Tensor:
        y = focus_transform(x)
        out = self.conv(y)

        return out


def focus_transform(x: Tensor) -> Tensor:
    """x(b,c,w,h) -> y(b,4c,w/2,h/2)"""
    y = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
    return y


def space_to_depth(x: Tensor) -> Tensor:
    """x(b,c,w,h) -> y(b,4c,w/2,h/2)"""
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


class TransformerLayer(nn.Module):
    """
    Transformer layer <https://arxiv.org/abs/2010.11929>.
    Remove the LayerNorm layers for better performance

    Args:
        c (int): number of channels
        num_heads: number of heads
    """

    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)

        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    """
    Vision Transformer <https://arxiv.org/abs/2010.11929>.

    Args:
        c1 (int): number of input channels
        c2 (int): number of output channels
        num_heads: number of heads
        num_layers: number of layers
    """

    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2, version="r4.0")
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
        return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e, version="r4.0")
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[GhostBottleneck(c_, c_) for _ in range(n)])


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),
        )  # pw-linear
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False))
            if s == 2
            else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()
        # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs.
    # Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class
    multi_label = False  # NMS multiple labels per box
    max_det = 1000  # maximum number of detections per image

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def autoshape(self):
        LOGGER.info("AutoShape already enabled, skipping... ")  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_sync()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != "cpu"):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (
            (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])
        )  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f"image{i}"  # filename
            if isinstance(im, (str, Path)):  # filename or uri
                im, f = (
                    Image.open(requests.get(im, stream=True).raw if str(im).startswith("http") else im),
                    im,
                )
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, "filename", f) or f
            files.append(Path(f).with_suffix(".jpg").name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = size / max(s)  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        # inference shape
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.0  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(enabled=p.device.type != "cpu"):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(
                y,
                self.conf,
                iou_thres=self.iou,
                classes=self.classes,
                multi_label=self.multi_label,
                max_det=self.max_det,
            )  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        # normalizations
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1.0, 1.0], device=d) for im in imgs]
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(
        self,
        pprint=False,
        show=False,
        save=False,
        crop=False,
        render=False,
        save_dir=Path(""),
    ):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f"image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} "
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    annotator = Annotator(im, pil=not self.ascii)
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f"{self.names[int(cls)]} {conf:.2f}"
                        if crop:
                            file = save_dir / "crops" / self.names[int(cls)] / self.files[i] if save else None
                            crops.append(
                                {
                                    "box": box,
                                    "conf": conf,
                                    "cls": cls,
                                    "label": label,
                                    "im": save_one_box(box, im, file=file, save=save),
                                }
                            )
                        else:  # all others
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im
            else:
                str += "(no detections)"

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                LOGGER.info(str.rstrip(", "))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f"Saved results to {save_dir}\n")
            return crops

    def print(self):
        self.display(pprint=True)  # print results
        LOGGER.info(
            f"Speed: {self.t[0]:.1f}ms pre-process, {self.t[1]:.1f}ms inference, "
            f"{self.t[2]:.1f}ms NMS per image at shape {tuple(self.s)}"
        )

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir="runs/detect/exp"):
        # increment save_dir
        save_dir = increment_path(save_dir, exist_ok=save_dir != "runs/detect/exp", mkdir=True)
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir="runs/detect/exp"):
        save_dir = (
            increment_path(save_dir, exist_ok=save_dir != "runs/detect/exp", mkdir=True) if save else None
        )
        return self.display(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = (
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "confidence",
            "class",
            "name",
        )  # xyxy columns
        cb = (
            "xcenter",
            "ycenter",
            "width",
            "height",
            "confidence",
            "class",
            "name",
        )  # xywh columns
        for k, c in zip(["xyxy", "xyxyn", "xywh", "xywhn"], [ca, ca, cb, cb]):
            # update
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ["imgs", "pred", "xyxy", "xyxyn", "xywh", "xywhn"]:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)
