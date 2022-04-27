from typing import Dict, Iterable, Callable

import torch
from torch import nn, Tensor


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, return_layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.return_layers = return_layers
        self._features = {layer: torch.empty(0) for layer in return_layers}

        for layer_id in return_layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output

        return fn

    def forward(self, images: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        _ = self.model(images, targets)
        return self._features
