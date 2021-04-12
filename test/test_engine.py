# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from pathlib import Path
import unittest

import torch
from torch import Tensor
import torch.utils.data
from torchvision.io import read_image

import pytorch_lightning as pl

from yolort.data import DetectionDataModule
from yolort.data.coco import CocoDetection
from yolort.data.transforms import collate_fn, default_train_transforms

from yolort.models.yolo import yolov5_darknet_pan_s_r31
from yolort.models.transform import nested_tensor_from_tensor_list
from yolort.models import yolov5s

from yolort.utils import prepare_coco128

from .dataset_utils import DummyCOCODetectionDataset

from typing import Dict


def default_loader(img_name, is_half=False):
    """
    Read Image using TorchVision.io Here
    """
    img = read_image(img_name)
    img = img.half() if is_half else img.float()  # uint8 to fp16/32
    img /= 255.  # 0 - 255 to 0.0 - 1.0

    return img


class EngineTester(unittest.TestCase):
    def test_train_with_vanilla_model(self):
        # Do forward over image
        img_name = "test/assets/zidane.jpg"
        img_tensor = default_loader(img_name)
        self.assertEqual(img_tensor.ndim, 3)
        # Add a dummy image to train
        img_dummy = torch.rand((3, 416, 360), dtype=torch.float32)

        images = nested_tensor_from_tensor_list([img_tensor, img_dummy])
        targets = torch.tensor([[0, 7, 0.3790, 0.5487, 0.3220, 0.2047],
                                [0, 2, 0.2680, 0.5386, 0.2200, 0.1779],
                                [0, 3, 0.1720, 0.5403, 0.1960, 0.1409],
                                [0, 4, 0.2240, 0.4547, 0.1520, 0.0705]], dtype=torch.float)

        model = yolov5_darknet_pan_s_r31(num_classes=12)
        model.train()
        out = model(images, targets)
        self.assertIsInstance(out, Dict)
        self.assertIsInstance(out["cls_logits"], Tensor)
        self.assertIsInstance(out["bbox_regression"], Tensor)
        self.assertIsInstance(out["objectness"], Tensor)

    def test_train_with_vanilla_module(self):
        """
        For issue #86: <https://github.com/zhiqwang/yolov5-rt-stack/issues/86>
        """
        # Define the device
        device = torch.device('cpu')

        # Prepare the datasets for training
        # Acquire the images and labels from the coco128 dataset
        data_path = Path('data-bin')
        coco128_dirname = 'coco128'
        coco128_path = data_path / coco128_dirname
        image_root = coco128_path / 'images' / 'train2017'
        annotation_file = coco128_path / 'annotations' / 'instances_train2017.json'

        if not annotation_file.is_file():
            prepare_coco128(data_path, dirname=coco128_dirname)

        batch_size = 4

        dataset = CocoDetection(image_root, annotation_file, default_train_transforms())
        sampler = torch.utils.data.RandomSampler(dataset)
        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=0)
        # Sample a pair of images/targets
        images, targets = next(iter(data_loader))
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Define the model
        model = yolov5s(num_classes=80)
        model.train()

        out = model(images, targets)
        self.assertIsInstance(out, Dict)
        self.assertIsInstance(out["cls_logits"], Tensor)
        self.assertIsInstance(out["bbox_regression"], Tensor)
        self.assertIsInstance(out["objectness"], Tensor)

    def test_train_one_epoch(self):
        # Setup the DataModule
        train_dataset = DummyCOCODetectionDataset(num_samples=128)
        data_module = DetectionDataModule(train_dataset, batch_size=16)
        # Load model
        model = yolov5s()
        model.train()
        # Trainer
        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(model, data_module)

    def test_test_dataloaders(self):
        # Config dataset
        num_samples = 128
        batch_size = 4
        # Setup the DataModule
        train_dataset = DummyCOCODetectionDataset(num_samples=num_samples)
        data_module = DetectionDataModule(train_dataset, batch_size=batch_size)
        # Load model
        model = yolov5s(pretrained=True)
        model.eval()
        # Trainer
        trainer = pl.Trainer(max_epochs=1)
        trainer.test(model, test_dataloaders=data_module.val_dataloader(batch_size=batch_size))

    def test_predict_with_vanilla_model(self):
        # Set image inputs
        img_name = "test/assets/zidane.jpg"
        img_input = default_loader(img_name)
        self.assertEqual(img_input.ndim, 3)
        # Load model
        model = yolov5s(pretrained=True)
        model.eval()
        # Perform inference on a list of tensors
        out = model([img_input])
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], Dict)
        self.assertIsInstance(out[0]["boxes"], Tensor)
        self.assertIsInstance(out[0]["labels"], Tensor)
        self.assertIsInstance(out[0]["scores"], Tensor)

    def test_predict_with_tensor(self):
        # Set image inputs
        img_name = "test/assets/zidane.jpg"
        img_tensor = default_loader(img_name)
        self.assertEqual(img_tensor.ndim, 3)
        # Load model
        model = yolov5s(pretrained=True)
        model.eval()
        # Perform inference on a list of image files
        predictions = model.predict(img_tensor)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 1)
        self.assertIsInstance(predictions[0], Dict)
        self.assertIsInstance(predictions[0]["boxes"], Tensor)
        self.assertIsInstance(predictions[0]["labels"], Tensor)
        self.assertIsInstance(predictions[0]["scores"], Tensor)

    def test_predict_with_tensors(self):
        # Set image inputs
        img_tensor1 = default_loader("test/assets/zidane.jpg")
        self.assertEqual(img_tensor1.ndim, 3)
        img_tensor2 = default_loader("test/assets/bus.jpg")
        self.assertEqual(img_tensor2.ndim, 3)
        img_tensors = [img_tensor1, img_tensor2]
        # Load model
        model = yolov5s(pretrained=True)
        model.eval()
        # Perform inference on a list of image files
        predictions = model.predict(img_tensors)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 2)
        self.assertIsInstance(predictions[0], Dict)
        self.assertIsInstance(predictions[0]["boxes"], Tensor)
        self.assertIsInstance(predictions[0]["labels"], Tensor)
        self.assertIsInstance(predictions[0]["scores"], Tensor)

    def test_predict_with_image_file(self):
        # Set image inputs
        img_name = "test/assets/zidane.jpg"
        # Load model
        model = yolov5s(pretrained=True)
        model.eval()
        # Perform inference on an image file
        predictions = model.predict(img_name)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 1)
        self.assertIsInstance(predictions[0], Dict)
        self.assertIsInstance(predictions[0]["boxes"], Tensor)
        self.assertIsInstance(predictions[0]["labels"], Tensor)
        self.assertIsInstance(predictions[0]["scores"], Tensor)

    def test_predict_with_image_files(self):
        # Set image inputs
        img_name1 = "test/assets/zidane.jpg"
        img_name2 = "test/assets/bus.jpg"
        img_names = [img_name1, img_name2]
        # Load model
        model = yolov5s(pretrained=True)
        model.eval()
        # Perform inference on a list of image files
        predictions = model.predict(img_names)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 2)
        self.assertIsInstance(predictions[0], Dict)
        self.assertIsInstance(predictions[0]["boxes"], Tensor)
        self.assertIsInstance(predictions[0]["labels"], Tensor)
        self.assertIsInstance(predictions[0]["scores"], Tensor)
