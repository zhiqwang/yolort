# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from pathlib import Path

import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from torch import Tensor
from torchvision.io import read_image

from pytorch_lightning import LightningDataModule

from .transforms import collate_fn, default_train_transforms, default_val_transforms
from .voc import VOCDetection
from .coco import CocoDetection
from .datapipeline import DataPipeline

from typing import Callable, List, Any, Optional, Type
from collections.abc import Sequence


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


class DetectionDataModule(LightningDataModule):
    """
    Wrapper of Datasets in LightningDataModule
    """
    def __init__(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._test_dataset = test_dataset

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self, batch_size: int = 16) -> None:
        """
        VOCDetection and CocoDetection
        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        # Creating data loaders
        sampler = torch.utils.data.RandomSampler(self._train_dataset)
        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)

        loader = DataLoader(
            self._train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

        return loader

    def val_dataloader(self, batch_size: int = 16) -> None:
        """
        VOCDetection and CocoDetection
        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        # Creating data loaders
        sampler = torch.utils.data.SequentialSampler(self._val_dataset)

        loader = DataLoader(
            self._val_dataset,
            batch_size,
            sampler=sampler,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

        return loader

    @property
    def data_pipeline(self) -> DataPipeline:
        if self._data_pipeline is None:
            self._data_pipeline = self.default_pipeline()
        return self._data_pipeline

    @data_pipeline.setter
    def data_pipeline(self, data_pipeline) -> None:
        self._data_pipeline = data_pipeline

    @staticmethod
    def default_pipeline() -> DataPipeline:
        return ObjectDetectionDataPipeline()


class VOCDetectionDataModule(DetectionDataModule):
    def __init__(
        self,
        data_path: str,
        years: List[str] = ["2007", "2012"],
        train_transform: Optional[Callable] = default_train_transforms,
        val_transform: Optional[Callable] = default_val_transforms,
        batch_size: int = 1,
        num_workers: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        train_dataset, num_classes = self.build_datasets(
            data_path, image_set='train', years=years, transforms=train_transform)
        val_dataset, _ = self.build_datasets(
            data_path, image_set='val', years=years, transforms=val_transform)

        super().__init__(train_dataset=train_dataset, val_dataset=val_dataset,
                         batch_size=batch_size, num_workers=num_workers, *args, **kwargs)

        self.num_classes = num_classes

    @staticmethod
    def build_datasets(data_path, image_set, years, transforms):
        datasets = []
        for year in years:
            dataset = VOCDetection(
                data_path,
                year=year,
                image_set=image_set,
                transforms=transforms(),
            )
            datasets.append(dataset)

        num_classes = len(datasets[0].prepare.CLASSES)

        if len(datasets) == 1:
            return datasets[0], num_classes
        else:
            return torch.utils.data.ConcatDataset(datasets), num_classes


class CocoDetectionDataModule(DetectionDataModule):
    def __init__(
        self,
        data_path: str,
        year: str = "2017",
        train_transform: Optional[Callable] = default_train_transforms,
        val_transform: Optional[Callable] = default_val_transforms,
        batch_size: int = 1,
        num_workers: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        train_dataset = self.build_datasets(
            data_path, image_set='train', year=year, transforms=train_transform)
        val_dataset = self.build_datasets(
            data_path, image_set='val', year=year, transforms=val_transform)

        super().__init__(train_dataset=train_dataset, val_dataset=val_dataset,
                         batch_size=batch_size, num_workers=num_workers, *args, **kwargs)

        self.num_classes = 80

    @staticmethod
    def build_datasets(data_path, image_set, year, transforms):
        ann_file = Path(data_path).joinpath('annotations').joinpath(f"instances_{image_set}{year}.json")
        return CocoDetection(data_path, ann_file, transforms())
