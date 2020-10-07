# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
import warnings

from collections import OrderedDict
from typing import List, Dict

import torch
from torch import nn, Tensor

from utils.general import check_anchor_order


class YOLO(nn.Module):
    def __init__(
        self,
        body: nn.Module,
        box_head: nn.Module,
        post_process: nn.Module,
        stride: List[float] = [8., 16., 32.],
    ):
        super().__init__()
        self.body = body
        box_head.stride = torch.tensor(stride)
        box_head.anchors /= box_head.stride.view(-1, 1, 1)
        check_anchor_order(box_head)
        self.box_head = box_head
        self.post_process = post_process
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, detections: Tensor, features: Tensor):
        if self.training:
            return features

        return detections

    def forward(self, samples):
        features = self.body(samples)
        detections = self.box_head(features)

        if not self.training:
            detections = self.post_process(detections)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("YOLO always returns a (detections, features) tuple in scripting")
                self._has_warned = True
            return (detections, features)
        else:
            return self.eager_outputs(detections, features)


class Body(nn.Module):
    def __init__(
        self,
        body: nn.Module,
        return_layers_body: dict,
    ):
        super().__init__()
        self.body = IntermediateLayerGetter(
            body.model,
            return_layers=return_layers_body,
            save_list=body.save,
        )

    def forward(self, inputs: Tensor):
        body = self.body(inputs)
        out: List[Tensor] = []

        for name, x in body.items():
            out.append(x)

        return out


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
