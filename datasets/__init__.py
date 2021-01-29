# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from models.transform import nested_tensor_from_tensor_list

from .coco import build as build_coco
from .voc import build as build_voc


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


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


def build_dataset(image_set, dataset_year, args):

    datasets = []
    for year in dataset_year:
        if args.dataset_file == 'coco':
            dataset = build_coco(image_set, year, args)
        elif args.dataset_file == 'voc':
            dataset = build_voc(image_set, year, args)
        else:
            raise ValueError(f'dataset {args.dataset_file} not supported')
        datasets.append(dataset)

    if len(datasets) == 1:
        return datasets[0]
    else:
        return torch.utils.data.ConcatDataset(datasets)
