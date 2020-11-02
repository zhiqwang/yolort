# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
import warnings
from collections import OrderedDict
from typing import Union

import torch
from torch import nn, Tensor

from torch.jit.annotations import Tuple, List, Dict, Optional

from utils.general import check_anchor_order


class YOLO(nn.Module):
    def __init__(
        self,
        body: nn.Module,
        box_head: nn.Module,
        post_process: nn.Module,
        transform: nn.Module,
        stride: List[float] = [8., 16., 32.],
    ):
        super().__init__()
        self.transform = transform
        self.body = body
        box_head.stride = torch.tensor(stride)
        box_head.anchors /= box_head.stride.view(-1, 1, 1)
        check_anchor_order(box_head)
        self.box_head = box_head
        self.post_process = post_process
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(
        self,
        features: List[Tensor],
        detections: List[Dict[str, Tensor]],
    ) -> Union[List[Tensor], List[Dict[str, Tensor]]]:
        if self.training:
            return features

        return detections

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Tensor], List[Dict[str, Tensor]]]:
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During Training, it returns a dict[Tensor] which contains the losses
                TODO, currently this repo doesn't support training.
                During Testing, it returns list[BoxList] contains additional fields
                like `scores` and `labels`.
        """
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        features = self.body(images.tensors)
        predictions = self.box_head(features)
        detections = self.post_process(predictions)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("YOLO always returns a (features, detections) tuple in scripting")
                self._has_warned = True
            return (features, detections)
        else:
            return self.eager_outputs(features, detections)

    def update_ultralytics(self, checkpoint_path_ultralytics: str):
        checkpoint = torch.load(checkpoint_path_ultralytics, map_location="cpu")
        state_dict = checkpoint['model'].float().state_dict()  # to FP32

        # Update body features
        for name, params in self.body.features.named_parameters(prefix='model'):
            params.data.copy_(state_dict[name])

        for name, buffers in self.body.features.named_buffers(prefix='model'):
            buffers.copy_(state_dict[name])

        # Update box heads
        for name, params in self.box_head.named_parameters(prefix='model.24'):
            params.data.copy_(state_dict[name])

        for name, buffers in self.box_head.named_buffers(prefix='model.24'):
            buffers.copy_(state_dict[name])


class YoloBody(nn.Module):
    def __init__(
        self,
        yolo_body: nn.Module,
        return_layers: dict,
    ):
        super().__init__()
        self.features = IntermediateLayerGetter(
            yolo_body.model,
            return_layers=return_layers,
            save_list=yolo_body.save,
        )

    def forward(self, inputs: Tensor):
        body = self.features(inputs)
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
