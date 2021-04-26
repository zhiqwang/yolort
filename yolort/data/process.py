from collections.abc import Sequence

from torch import Tensor
from torchvision import transforms as T

from pytorch_lightning.utilities.exceptions import MisconfigurationException
from flash.data.utils import _contains_any_tensor
from flash.data.auto_dataset import AutoDataset
from flash.data.process import DefaultPreprocess, Preprocess

from .transforms import collate_fn
from .coco import COCODetection
from ..utils.image_utils import read_image_to_tensor

from typing import Any


class ObjectDetectionPreprocess(DefaultPreprocess):

    to_tensor = T.ToTensor()

    def load_data(self, metadata: Any, dataset: AutoDataset) -> COCODetection:
        # Extract folder, coco annotation file and the transform to be applied on the images
        folder, ann_file, transform = metadata
        ds = COCODetection(folder, ann_file, transform)
        if self.training:
            dataset.num_classes = ds.num_classes
        return ds

    def predict_load_data(self, samples):
        return samples

    def pre_tensor_transform(self, samples: Any) -> Any:
        if _contains_any_tensor(samples):
            return samples

        if isinstance(samples, str):
            samples = [samples]

        if isinstance(samples, (list, tuple)) and all(isinstance(p, str) for p in samples):
            outputs = []
            for sample in samples:
                outputs.append(read_image_to_tensor(sample))
            return outputs
        raise MisconfigurationException("The samples should either be a tensor, a list of paths or a path.")

    def to_tensor_transform(self, sample) -> Any:
        return self.to_tensor(sample[0]), sample[1]

    def predict_to_tensor_transform(self, sample) -> Any:
        return self.to_tensor(sample[0])

    def collate(self, samples: Any) -> Any:
        if not isinstance(samples, Tensor):
            elem = samples[0]
            if isinstance(elem, Sequence):
                return tuple(zip(*samples))
            return collate_fn(samples)
        return samples.unsqueeze(dim=0)
