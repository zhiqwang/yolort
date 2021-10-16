# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Activation functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    """
    Export-friendly version of nn.SiLU(). Starting with PyTorch 1.8,
    this operator supports exporting to ONNX, and there is also a
    build-in implementation of it on TVM.

    Ref: <https://arxiv.org/pdf/1606.08415.pdf>
    """

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Hardswish(nn.Module):
    """
    Export-friendly version of nn.Hardswish(). Starting with PyTorch 1.8,
    this operator supports exporting to ONNX, and currently this module
    is only used for TVM.
    """

    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0
