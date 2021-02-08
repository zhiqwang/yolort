# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from pathlib import Path

import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from pytorch_lightning import LightningDataModule

from .transforms import collate_fn, default_train_transforms, default_val_transforms
from .voc import VOCDetection
from .coco import CocoDetection

from typing import Callable, List, Any, Optional


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
