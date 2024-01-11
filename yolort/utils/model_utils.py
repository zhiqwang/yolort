#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import contextlib
from copy import deepcopy
from typing import Sequence

import torch
import torch.nn as nn

__all__ = [
    "get_model_info",
    "adjust_status",
]


def get_model_info(model: nn.Module, tsize: Sequence[int]) -> str:
    from thop import profile

    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info


@contextlib.contextmanager
def adjust_status(module: nn.Module, training: bool = False) -> nn.Module:
    """Adjust module to training/eval mode temporarily.

    Args:
        module (nn.Module): module to adjust status.
        training (bool): training mode to set. True for train mode, False fro eval mode.

    Examples:
        >>> with adjust_status(model, training=False):
        ...     model(data)
    """
    status = {}

    def backup_status(module):
        for m in module.modules():
            # save prev status to dict
            status[m] = m.training
            m.training = training

    def recover_status(module):
        for m in module.modules():
            # recover prev status from dict
            m.training = status.pop(m)

    backup_status(module)
    yield module
    recover_status(module)
