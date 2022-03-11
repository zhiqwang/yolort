# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from yolort.data import contains_any_tensor, _helper as data_helper
from yolort.data.data_module import DetectionDataModule


def test_contains_any_tensor():
    dummy_numpy = np.random.randn(3, 6)
    assert not contains_any_tensor(dummy_numpy)
    dummy_tensor = torch.rand(3, 6)
    assert contains_any_tensor(dummy_tensor)
    dummy_tensors = [torch.rand(3, 6), torch.rand(9, 5)]
    assert contains_any_tensor(dummy_tensors)


def test_get_dataset():
    # Acquire the images and labels from the coco128 dataset
    train_dataset = data_helper.get_dataset(data_root="data-bin", mode="train")
    # Test the datasets
    image, target = next(iter(train_dataset))
    assert isinstance(image, Tensor)
    assert isinstance(target, dict)


def test_get_dataloader():
    batch_size = 8
    data_loader = data_helper.get_dataloader(data_root="data-bin", mode="train", batch_size=batch_size)
    # Test the dataloader
    images, targets = next(iter(data_loader))

    assert len(images) == batch_size
    assert isinstance(images[0], Tensor)
    assert len(images[0]) == 3
    assert len(targets) == batch_size
    assert isinstance(targets[0], dict)
    assert isinstance(targets[0]["image_id"], Tensor)
    assert isinstance(targets[0]["boxes"], Tensor)
    assert isinstance(targets[0]["labels"], Tensor)
    assert isinstance(targets[0]["orig_size"], Tensor)


def test_detection_data_module():
    # Setup the DataModule
    batch_size = 4
    train_dataset = data_helper.get_dataset(data_root="data-bin", mode="train")
    data_module = DetectionDataModule(train_dataset, batch_size=batch_size)
    assert data_module.batch_size == batch_size

    data_loader = data_module.train_dataloader()
    images, targets = next(iter(data_loader))
    assert len(images) == batch_size
    assert isinstance(images[0], Tensor)
    assert len(images[0]) == 3
    assert len(targets) == batch_size
    assert isinstance(targets[0], dict)
    assert isinstance(targets[0]["image_id"], Tensor)
    assert isinstance(targets[0]["boxes"], Tensor)
    assert isinstance(targets[0]["labels"], Tensor)


def test_prepare_coco128():
    data_path = Path("data-bin")
    coco128_dirname = "coco128"
    data_helper.prepare_coco128(data_path, dirname=coco128_dirname)
    annotation_file = data_path / coco128_dirname / "annotations" / "instances_train2017.json"
    assert annotation_file.is_file()
