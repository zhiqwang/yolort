# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from pathlib import Path
import unittest

import torch

from yolort.data.coco import CocoDetection
from yolort.data.transforms import collate_fn, default_train_transforms
from yolort.utils import prepare_coco128

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
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(target, Dict)

        batch_size = 4
        sampler = torch.utils.data.RandomSampler(dataset)
        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=0)
        # Test the dataloader
        images, targets = next(iter(loader))

        self.assertEqual(len(images), batch_size)
        self.assertEqual(len(targets), batch_size)

    def test_prepare_coco128(self):
        data_path = Path('data-bin')
        coco128_dirname = 'coco128'
        prepare_coco128(data_path, dirname=coco128_dirname)
        annotation_file = data_path / coco128_dirname / 'annotations' / 'instances_train2017.json'
        self.assertTrue(annotation_file.is_file())
