# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from typing import Dict, Tuple, Callable

import torch
from torch import nn, Tensor


def dict_to_tuple(out_dict: Dict[str, Tensor]) -> Tuple:
    """
    Convert the model output dictionary to tuple format.
    """
    if "masks" in out_dict.keys():
        return (
            out_dict["boxes"],
            out_dict["scores"],
            out_dict["labels"],
            out_dict["masks"],
        )
    return out_dict["boxes"], out_dict["scores"], out_dict["labels"]


class TraceWrapper(nn.Module):
    """
    This is a wrapper for `torch.jit.trace`, as there are some scenarios
    where `torch.jit.script` support is limited.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return dict_to_tuple(out[0])


@torch.no_grad()
def get_trace_module(
    model_func: Callable[..., nn.Module],
    input_shape: Tuple[int, int] = (416, 416),
):
    """
    Get the tracing of a given model function.

    Example:

        >>> from yolort.models import yolov5s
        >>> from yolort.relaying.trace_wrapper import get_trace_module
        >>>
        >>> model = yolov5s(pretrained=True)
        >>> tracing_module = get_trace_module(model)
        >>> print(tracing_module.code)
        def forward(self,
            x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
          _0, _1, _2, = (self.model).forward(x, )
          return (_0, _1, _2)

    Args:
        model_func (Callable): The model function to be traced.
        input_shape (Tuple[int, int]): Shape size of the input image.
    """
    model = TraceWrapper(model_func)
    model.eval()

    dummy_input = torch.rand(1, 3, *input_shape)
    trace_module = torch.jit.trace(model, dummy_input)
    trace_module.eval()

    return trace_module
