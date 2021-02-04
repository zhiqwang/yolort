# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import torch.utils.data
from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule

from models.transform import nested_tensor_from_tensor_list
from .coco import build as build_coco
from .voc import build as build_voc

from typing import List, Any


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


def build_dataset(data_path, dataset_type, image_set, dataset_year):

    datasets = []
    for year in dataset_year:
        if dataset_type == 'coco':
            dataset = build_coco(data_path, image_set, year)
        elif dataset_type == 'voc':
            dataset = build_voc(data_path, image_set, year)
        else:
            raise ValueError(f'dataset {dataset_type} not supported')
        datasets.append(dataset)

    if len(datasets) == 1:
        return datasets[0]
    else:
        return torch.utils.data.ConcatDataset(datasets)


class DetectionDataModule(LightningDataModule):
    """
    Wrapper of Datasets in LightningDataModule
    """
    def __init__(
        self,
        data_path: str,
        dataset_type: str,
        dataset_year: List[str],
        num_workers: int = 4,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.data_path = data_path
        self.dataset_type = dataset_type
        self.dataset_year = dataset_year
        self.num_workers = num_workers

    def train_dataloader(self, batch_size: int = 16) -> None:
        """
        VOCDetection and CocoDetection
        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        dataset = build_dataset(self.data_path, self.dataset_type, 'train', self.dataset_year)

        # Creating data loaders
        sampler = torch.utils.data.RandomSampler(dataset)
        batch_sampler = torch.utils.data.BatchSampler(
            sampler, batch_size, drop_last=True,
        )

        loader = DataLoader(
            dataset,
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
        dataset = build_dataset(self.data_path, self.dataset_type, 'val', self.dataset_year)

        # Creating data loaders
        sampler = torch.utils.data.SequentialSampler(dataset)

        loader = DataLoader(
            dataset,
            batch_size,
            sampler=sampler,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

        return loader
