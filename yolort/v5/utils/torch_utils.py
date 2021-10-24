# YOLOv5 by Ultralytics, GPL-3.0 license
"""
PyTorch utils
"""

import datetime
import logging
import math
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training
    wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])


def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def date_modified(path=__file__):
    # return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f"{t.year}-{t.month}-{t.day}"


def git_describe(path=Path(__file__).parent):
    # path must be a directory
    # return human-readable git description,
    # i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    s = f"git -C {path} describe --tags --long --always"
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError:
        return ""  # not a git repository


def select_device(device="", batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f"YOLOv5 {git_describe() or date_modified()} torch {torch.__version__} "  # string
    device = str(device).strip().lower().replace("cuda:", "")  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    if cpu:
        # force torch.cuda.is_available() = False
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device:  # non-cpu device requested
        # set environment variable
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        # check availability
        assert torch.cuda.is_available(), f"CUDA unavailable, invalid device {device} requested"

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        devices = device.split(",") if device else "0"
        n = len(devices)  # device count
        if n > 1 and batch_size:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f"batch-size {batch_size} not multiple of GPU count {n}"
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            # bytes to MB
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"
    else:
        s += "CPU\n"

    # emoji-safe
    LOGGER.info(s.encode().decode("ascii", "ignore") if platform.system() == "Windows" else s)
    return torch.device("cuda:0" if cuda else "cpu")


def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def profile(input, ops, n=10, device=None):
    # YOLOv5 speed/memory/FLOPs profiler
    #
    # Usage:
    #     input = torch.randn(16, 3, 640, 640)
    #     m1 = lambda x: x * torch.sigmoid(x)
    #     m2 = nn.SiLU()
    #     profile(input, [m1, m2], n=100)  # profile over 100 iterations

    results = []
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    device = device or select_device()
    print(
        f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}"
        f"{'forward (ms)':>14s}{'backward (ms)':>14s}"
        f"{'input':>24s}{'output':>24s}"
    )

    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, "to") else m  # device
            m = (
                m.half()
                if hasattr(m, "half") and isinstance(x, torch.Tensor) and x.dtype is torch.float16
                else m
            )
            # dt forward, backward
            tf, tb, t = 0.0, 0.0, [0.0, 0.0, 0.0]
            if thop is None:
                flops = 0
            else:
                # GFLOPs
                flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1e9 * 2

            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    t[1] = time_sync()
                    try:
                        _ = (sum([yi.sum() for yi in y]) if isinstance(y, list) else y).sum().backward()
                        t[2] = time_sync()
                    # no backward method
                    except Exception as e:
                        print(e)
                        t[2] = float("nan")
                    # ms per op forward
                    tf += (t[1] - t[0]) * 1000 / n
                    # ms per op backward
                    tb += (t[2] - t[1]) * 1000 / n
                mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0  # (GB)
                s_in = tuple(x.shape) if isinstance(x, torch.Tensor) else "list"
                s_out = tuple(y.shape) if isinstance(y, torch.Tensor) else "list"
                # parameters
                p = sum(list(x.numel() for x in m.parameters())) if isinstance(m, nn.Module) else 0
                print(
                    f"{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}"
                    f"{str(s_in):>24s}{str(s_out):>24s}"
                )
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                print(e)
                results.append(None)
            torch.cuda.empty_cache()
    return results


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )


def de_parallel(model):
    # De-parallelize a model:
    # returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes,
    # omitting 'exclude' keys, using da values
    return {
        k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape
    }


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    # Finds layer indices matching module class 'mclass'
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def sparsity(model):
    # Return global model sparsity
    a, b = 0.0, 0.0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    import torch.nn.utils.prune as prune

    print("Pruning model... ", end="")
    for _, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            # prune
            prune.l1_unstructured(m, name="weight", amount=amount)
            prune.remove(m, "weight")  # make permanent
    print(" %.3g global sparsity" % sparsity(model))


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def model_info(model, verbose=False, img_size=640):
    # Model information. img_size may be int or list,
    # i.e. img_size=640 or img_size=[640, 320]

    # number parameters
    n_p = sum(x.numel() for x in model.parameters())
    # number gradients
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)
    if verbose:
        print(
            "%5s %40s %9s %12s %20s %10s %10s"
            % ("layer", "name", "gradient", "parameters", "shape", "mu", "sigma")
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            print(
                f"{i:5g} {name:40s} {p.requires_grad:9s} {p.numel():12g} "
                f"{list(p.shape):20s} {p.mean():10.3g} {p.std():10.3g}"
            )

    try:  # FLOPs
        from thop import profile

        stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32
        # input
        img = torch.zeros(
            (1, model.yaml.get("ch", 3), stride, stride),
            device=next(model.parameters()).device,
        )
        # stride GFLOPs
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1e9 * 2
        # expand if int/float
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]
        # 640x640 GFLOPs
        fs = ", %.1f GFLOPs" % (flops * img_size[0] / stride * img_size[1] / stride)
    except (ImportError, Exception):
        fs = ""

    LOGGER.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def load_classifier(name="resnet101", n=2):
    # Loads a pretrained model reshaped to n-class output
    model = torchvision.models.__dict__[name](pretrained=True)

    # ResNet model properties
    # input_size = [3, 224, 224]
    # input_space = 'RGB'
    # input_range = [0, 1]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # Reshape output to n classes
    filters = model.fc.weight.shape[1]
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n
    return model


def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    # img(16,3,256,416)
    # scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        # new size
        s = (int(h * ratio), int(w * ratio))
        # resize
        img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)
        # pad/crop img
        if not same_shape:
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        # value = imagenet mean
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


class EarlyStopping:
    # YOLOv5 simple early stopper
    def __init__(self, patience=30):
        # i.e. mAP
        self.best_fitness = 0.0
        self.best_epoch = 0
        # epochs to wait after fitness stops improving to stop
        self.patience = patience

    def __call__(self, epoch, fitness):
        # >= 0 to allow for early zero-fitness stage of training
        if fitness >= self.best_fitness:
            self.best_epoch = epoch
            self.best_fitness = fitness
        # stop training if patience exceeded
        stop = (epoch - self.best_epoch) >= self.patience
        if stop:
            LOGGER.info(f"EarlyStopping patience {self.patience} exceeded, stopping training.")
        return stop


class ModelEMA:
    """
    Model Exponential Moving Average from
    https://github.com/rwightman/pytorch-image-models

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        # FP32 EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            # model state_dict
            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
