# Copyright (c) 2021, yolort team. All rights reserved.

import logging
from pathlib import Path, PosixPath
from zipfile import ZipFile

import torch
from tabulate import tabulate

from .transforms import collate_fn


def create_small_table(small_dict):
    """
    Create a small table using the keys of small_dict as headers. This is only
    suitable for small dictionaries. Copy from:
    <https://github.com/facebookresearch/detectron2/blob/7205996/detectron2/utils/logger.py#L209>

    Args:
        small_dict (dict): a result dictionary of only a few items.

    Returns:
        str: the table as a string.
    """
    keys, values = tuple(zip(*small_dict.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    return table


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
