# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from pathlib import Path

import numpy as np
import pytest
import torch

from torch import Tensor
from yolort.exp import Exp
from yolort.data import DataPrefetcher
from yolort.utils import contains_any_tensor
from torch import distributed as dist

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def test_contains_any_tensor():
    dummy_numpy = np.random.randn(3, 6)
    assert not contains_any_tensor(dummy_numpy)
    dummy_tensor = torch.rand(3, 6)
    assert contains_any_tensor(dummy_tensor)
    dummy_tensors = [torch.rand(3, 6), torch.rand(9, 5)]
    assert contains_any_tensor(dummy_tensors)


def test_get_dataset():
    # Acquire the images and labels from the coco128 dataset
    train_dataset = Exp().get_dataset(cache=True)
    # Test the datasets
    image, target, _, _ = next(iter(train_dataset))
    assert image.shape == (3, 640, 640)
    assert target.shape ==(50, 5)

def test_get_dataloader():
    batch_size = 8
    is_distributed = get_world_size() > 1
    data_loader = Exp().get_data_loader(
            batch_size=batch_size,
            is_distributed=is_distributed,
            no_aug=False,
            cache_img=None,
        )
    prefetcher = DataPrefetcher(data_loader)
    images, targets = prefetcher.next()

    assert len(images) == batch_size
    assert isinstance(images[0], Tensor)
    assert len(images[0]) == 3
    assert len(targets) == batch_size
    assert isinstance(targets[0], Tensor)

@pytest.mark.skip("Remove Lightning dependency")
def test_detection_data_module():
    from yolort.data.data_module import DetectionDataModule

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

