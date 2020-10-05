from collections import OrderedDict
from typing import List, Dict

import torch
from torch import nn, Tensor


class YOLO(nn.Module):
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
