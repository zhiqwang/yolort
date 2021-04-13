# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import torch

from .coco import COCODetection


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, COCODetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, COCODetection):
        return dataset.coco
    else:
        raise NotImplementedError("Currently only supports COCO datasets")
