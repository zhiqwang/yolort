# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
import warnings
from typing import Union

import torch
from torch import nn, Tensor

from torch.jit.annotations import Tuple, List, Dict, Optional


class YOLO(nn.Module):
    def __init__(
        self,
        body: nn.Module,
        box_head: nn.Module,
        post_process: nn.Module,
        transform: nn.Module,
    ):
        super().__init__()
        self.transform = transform
        self.body = body
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
