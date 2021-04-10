# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from pathlib import Path
import unittest

import torch.utils.data
from torch import Tensor

from yolort.data import DetectionDataModule
from yolort.data.coco import CocoDetection
from yolort.data.transforms import collate_fn, default_train_transforms
from yolort.utils import prepare_coco128

from .dataset_utils import DummyCOCODetectionDataset

from typing import Dict


class DataPipelineTester(unittest.TestCase):
    def test_vanilla_dataloader(self):
        # Acquire the images and labels from the coco128 dataset
        data_path = Path('data-bin')
        coco128_dirname = 'coco128'
        coco128_path = data_path / coco128_dirname
        image_root = coco128_path / 'images' / 'train2017'
        annotation_file = coco128_path / 'annotations' / 'instances_train2017.json'

        if not annotation_file.is_file():
            prepare_coco128(data_path, dirname=coco128_dirname)

        dataset = CocoDetection(image_root, annotation_file, default_train_transforms())
        # Test the datasets
        image, target = next(iter(dataset))
        self.assertIsInstance(image, Tensor)
        self.assertIsInstance(target, Dict)

        batch_size = 4
        sampler = torch.utils.data.RandomSampler(dataset)
        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=0)
        # Test the dataloader
        images, targets = next(iter(data_loader))

        self.assertEqual(len(images), batch_size)
        self.assertIsInstance(images[0], Tensor)
        self.assertEqual(len(images[0]), 3)
        self.assertEqual(len(targets), batch_size)
        self.assertIsInstance(targets[0], Dict)
        self.assertIsInstance(targets[0]["image_id"], Tensor)
        self.assertIsInstance(targets[0]["boxes"], Tensor)
        self.assertIsInstance(targets[0]["labels"], Tensor)
        self.assertIsInstance(targets[0]["orig_size"], Tensor)

    def test_detection_data_module(self):
        # Setup the DataModule
        batch_size = 4
        train_dataset = DummyCOCODetectionDataset(num_samples=128)
        data_module = DetectionDataModule(train_dataset, batch_size=batch_size)
        self.assertEqual(data_module.batch_size, batch_size)

        data_loader = data_module.train_dataloader(batch_size=batch_size)
        images, targets = next(iter(data_loader))
        self.assertEqual(len(images), batch_size)
        self.assertIsInstance(images[0], Tensor)
        self.assertEqual(len(images[0]), 3)
        self.assertEqual(len(targets), batch_size)
        self.assertIsInstance(targets[0], Dict)
        self.assertIsInstance(targets[0]["image_id"], Tensor)
        self.assertIsInstance(targets[0]["boxes"], Tensor)
        self.assertIsInstance(targets[0]["labels"], Tensor)

    def test_prepare_coco128(self):
        data_path = Path('data-bin')
        coco128_dirname = 'coco128'
        prepare_coco128(data_path, dirname=coco128_dirname)
        annotation_file = data_path / coco128_dirname / 'annotations' / 'instances_train2017.json'
        self.assertTrue(annotation_file.is_file())
