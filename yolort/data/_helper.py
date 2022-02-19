# Copyright (c) 2021, yolort team. All rights reserved.

import logging
from pathlib import Path, PosixPath
from typing import Type, Any
from zipfile import ZipFile

import torch
from tabulate import tabulate
from torch import Tensor

from .coco import COCODetection
from .transforms import collate_fn, default_train_transforms, default_val_transforms


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


def contains_any_tensor(value: Any, dtype: Type = Tensor) -> bool:
    """
    Determine whether or not a list contains any Type
    """
    if isinstance(value, dtype):
        return True
    if isinstance(value, (list, tuple)):
        return any(contains_any_tensor(v, dtype=dtype) for v in value)
    elif isinstance(value, dict):
        return any(contains_any_tensor(v, dtype=dtype) for v in value.values())
    return False


def prepare_coco128(
    data_path: PosixPath,
    dirname: str = "coco128",
) -> None:
    """
    Prepare coco128 dataset to test.

    Args:
        data_path (PosixPath): root path of coco128 dataset.
        dirname (str): the directory name of coco128 dataset. Default: 'coco128'.
    """
    logger = logging.getLogger(__name__)

    if not data_path.is_dir():
        logger.info(f"Create a new directory: {data_path}")
        data_path.mkdir(parents=True, exist_ok=True)

    zip_path = data_path / "coco128.zip"
    coco128_url = "https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0/coco128.zip"
    if not zip_path.is_file():
        logger.info(f"Downloading coco128 datasets form {coco128_url}")
        torch.hub.download_url_to_file(coco128_url, zip_path, hash_prefix="a67d2887")

    coco128_path = data_path / dirname
    if not coco128_path.is_dir():
        logger.info(f"Unzipping dataset to {coco128_path}")
        with ZipFile(zip_path, "r") as zip_obj:
            zip_obj.extractall(data_path)


def get_dataset(data_root: str, mode: str = "val"):
    # Acquire the images and labels from the coco128 dataset
    data_path = Path(data_root)
    coco128_dirname = "coco128"
    coco128_path = data_path / coco128_dirname
    image_root = coco128_path / "images" / "train2017"
    annotation_file = coco128_path / "annotations" / "instances_train2017.json"

    if not annotation_file.is_file():
        prepare_coco128(data_path, dirname=coco128_dirname)

    if mode == "train":
        dataset = COCODetection(image_root, annotation_file, default_train_transforms())
    elif mode == "val":
        dataset = COCODetection(image_root, annotation_file, default_val_transforms())
    else:
        raise NotImplementedError(f"Currently not supports mode {mode}")

    return dataset


def get_dataloader(data_root: str, mode: str = "val", batch_size: int = 4):
    # Prepare the datasets for training
    # Acquire the images and labels from the coco128 dataset
    dataset = get_dataset(data_root=data_root, mode=mode)

    # We adopt the sequential sampler in order to repeat the experiment
    sampler = torch.utils.data.SequentialSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size,
        sampler=sampler,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    return loader
