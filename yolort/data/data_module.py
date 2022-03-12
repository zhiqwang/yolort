# Copyright (c) 2021, yolort team. All rights reserved.

from pathlib import Path
from typing import Callable, List, Any, Optional

import torch.utils.data
from torch.utils.data.dataset import Dataset

try:
    from pytorch_lightning import LightningDataModule
except ImportError:
    LightningDataModule = None

from .coco import COCODetection
from .transforms import collate_fn, default_train_transforms, default_val_transforms
from .voc import VOCDetection


class DetectionDataModule(LightningDataModule):
    """
    Wrapper of Datasets in LightningDataModule
    """

    def __init__(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        batch_size: int = 16,
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

    def train_dataloader(self) -> None:
        """
        VOCDetection and COCODetection
        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        # Creating data loaders
        batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(self._train_dataset),
            self.batch_size,
            drop_last=True,
        )

        loader = torch.utils.data.DataLoader(
            self._train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

        return loader

    def val_dataloader(self) -> None:
        """
        VOCDetection and COCODetection
        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        # Creating data loaders
        sampler = torch.utils.data.SequentialSampler(self._val_dataset)

        loader = torch.utils.data.DataLoader(
            self._val_dataset,
            self.batch_size,
            sampler=sampler,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

        return loader


class COCODetectionDataModule(DetectionDataModule):
    def __init__(
        self,
        data_path: str,
        anno_path: Optional[str] = None,
        num_classes: int = 80,
        data_task: str = "instances",
        train_set: str = "train2017",
        val_set: str = "val2017",
        skip_train_set: bool = False,
        skip_val_set: bool = False,
        train_transform: Optional[Callable] = default_train_transforms,
        val_transform: Optional[Callable] = default_val_transforms,
        batch_size: int = 1,
        num_workers: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        anno_path = Path(anno_path) if anno_path else Path(data_path) / "annotations"
        train_ann_file = anno_path / f"{data_task}_{train_set}.json"
        val_ann_file = anno_path / f"{data_task}_{val_set}.json"

        train_dataset = (
            None if skip_train_set else COCODetection(data_path, train_ann_file, train_transform())
        )
        val_dataset = None if skip_val_set else COCODetection(data_path, val_ann_file, val_transform())

        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            *args,
            **kwargs,
        )

        self.num_classes = num_classes


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
            data_path, image_set="train", years=years, transforms=train_transform
        )
        val_dataset, _ = self.build_datasets(
            data_path, image_set="val", years=years, transforms=val_transform
        )

        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            *args,
            **kwargs,
        )

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
