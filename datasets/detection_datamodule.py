# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from pytorch_lightning import LightningDataModule

from . import transforms as T
from models.transform import nested_tensor_from_tensor_list

from typing import List, Any, Optional


def collate_fn(batch):
    batch = list(zip(*batch))
    samples = nested_tensor_from_tensor_list(batch[0])

    targets = []
    for i, target in enumerate(batch[1]):
        num_objects = len(target['labels'])
        if num_objects > 0:
            targets_merged = torch.full((num_objects, 6), i, dtype=torch.float32)
            targets_merged[:, 1] = target['labels']
            targets_merged[:, 2:] = target['boxes']
            targets.append(targets_merged)
    targets = torch.cat(targets, dim=0)

    return samples, targets


def default_train_transforms():
    scales = [384, 416, 448, 480, 512, 544, 576, 608, 640, 672]
    scales_for_training = [(640, 640)]

    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomSelect(
            T.RandomResize(scales_for_training),
            T.Compose([
                T.RandomResize(scales),
                T.RandomSizeCrop(384, 480),
                T.RandomResize(scales_for_training),
            ])
        ),
        T.Compose([T.ToTensor(), T.Normalize()]),
    ])


def default_val_transforms():
    return T.Compose([T.Compose([T.ToTensor(), T.Normalize()])])


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
