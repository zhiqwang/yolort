# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from typing import Callable, Any, Optional, Type
from collections.abc import Sequence

from torch import Tensor
from torchvision.io import read_image

from .transforms import collate_fn
from .data_pipeline import DataPipeline


class ObjectDetectionDataPipeline(DataPipeline):
    """
    Modified from:
    <https://github.com/PyTorchLightning/lightning-flash/blob/24c5b66/flash/vision/detection/data.py#L133-L160>
    """
    def __init__(self, loader: Optional[Callable] = None):
        if loader is None:
            loader = lambda x: read_image(x) / 255.
        self._loader = loader

    def before_collate(self, samples: Any) -> Any:
        if _contains_any_tensor(samples, Tensor):
            return samples

        if isinstance(samples, str):
            samples = [samples]

        if isinstance(samples, (list, tuple)) and all(isinstance(p, str) for p in samples):
            outputs = []
            for sample in samples:
                output = self._loader(sample)
                outputs.append(output)
            return outputs

        raise NotImplementedError("The samples should either be a tensor or path, a list of paths or tensors.")

    def collate(self, samples: Any) -> Any:
        if not isinstance(samples, Tensor):
            elem = samples[0]

            if isinstance(elem, Sequence):
                return collate_fn(samples)

            return list(samples)

        return samples.unsqueeze(dim=0)

    def after_collate(self, batch: Any) -> Any:
        return (batch["x"], batch["target"]) if isinstance(batch, dict) else (batch, None)


def _contains_any_tensor(value: Any, dtype: Type = Tensor) -> bool:
    """
    TODO: we should refactor FlashDatasetFolder to better integrate
    with DataPipeline. That way, we wouldn't need this check.
    This is because we are running transforms in both places.

    Ref:
    <https://github.com/PyTorchLightning/lightning-flash/blob/24c5b66/flash/core/data/utils.py#L80-L90>
    """
    if isinstance(value, dtype):
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_any_tensor(v, dtype=dtype) for v in value)
    elif isinstance(value, dict):
        return any(_contains_any_tensor(v, dtype=dtype) for v in value.values())
    return False
