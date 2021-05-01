# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from pathlib import Path
import unittest
import numpy as np

import torch
from torch import Tensor

from yolort.data import DetectionDataModule, contains_any_tensor, _helper as data_helper

from typing import Dict


class DataPipelineTester(unittest.TestCase):
    def test_contains_any_tensor(self):
        dummy_numpy = np.random.randn(3, 6)
        self.assertFalse(contains_any_tensor(dummy_numpy))
        dummy_tensor = torch.randn(3, 6)
        self.assertTrue(contains_any_tensor(dummy_tensor))
        dummy_tensors = [torch.randn(3, 6), torch.randn(9, 5)]
        self.assertTrue(contains_any_tensor(dummy_tensors))

    def test_get_dataset(self):
        # Acquire the images and labels from the coco128 dataset
        train_dataset = data_helper.get_dataset(data_root='data-bin', mode='train')
        # Test the datasets
        image, target = next(iter(train_dataset))
        self.assertIsInstance(image, Tensor)
        self.assertIsInstance(target, Dict)

    def test_get_dataloader(self):
        batch_size = 8
        data_loader = data_helper.get_dataloader(data_root='data-bin', mode='train', batch_size=batch_size)
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
        train_dataset = data_helper.get_dataset(data_root='data-bin', mode='train')
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
        data_helper.prepare_coco128(data_path, dirname=coco128_dirname)
        annotation_file = data_path / coco128_dirname / 'annotations' / 'instances_train2017.json'
        self.assertTrue(annotation_file.is_file())
