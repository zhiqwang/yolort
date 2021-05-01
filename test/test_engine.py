# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import unittest
from pathlib import Path

import torch
from torch import Tensor
from torchvision.io import read_image

import pytorch_lightning as pl

from yolort.data import COCOEvaluator, DetectionDataModule, _helper as data_helper

from yolort.models import yolov5s
from yolort.models.yolo import yolov5_darknet_pan_s_r31
from yolort.models.transform import nested_tensor_from_tensor_list

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

        train_dataloader = data_helper.get_dataloader(data_root='data-bin', mode='train')
        # Sample a pair of images/targets
        images, targets = next(iter(train_dataloader))
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

    def test_training_step(self):
        # Setup the DataModule
        data_path = 'data-bin'
        train_dataset = data_helper.get_dataset(data_root=data_path, mode='train')
        val_dataset = data_helper.get_dataset(data_root=data_path, mode='val')
        data_module = DetectionDataModule(train_dataset, val_dataset, batch_size=16)
        # Load model
        model = yolov5s()
        model.train()
        # Trainer
        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(model, data_module)

    def test_vanilla_coco_evaluator(self):
        # Acquire the images and labels from the coco128 dataset
        val_dataloader = data_helper.get_dataloader(data_root='data-bin', mode='val')
        coco = data_helper.get_coco_api_from_dataset(val_dataloader.dataset)
        coco_evaluator = COCOEvaluator(coco)
        # Load model
        model = yolov5s(pretrained=True, score_thresh=0.001)
        model.eval()
        for images, targets in val_dataloader:
            preds = model(images)
            coco_evaluator.update(preds, targets)

        results = coco_evaluator.compute()
        self.assertGreater(results['AP'], 41.5)
        self.assertGreater(results['AP50'], 62.0)

    def test_test_epoch_end(self):
        # Acquire the annotation file
        data_path = Path('data-bin')
        coco128_dirname = 'coco128'
        data_helper.prepare_coco128(data_path, dirname=coco128_dirname)
        annotation_file = data_path / coco128_dirname / 'annotations' / 'instances_train2017.json'

        # Get dataloader to test
        val_dataloader = data_helper.get_dataloader(data_root=data_path, mode='val')

        # Load model
        model = yolov5s(pretrained=True, score_thresh=0.001, annotation_path=annotation_file)

        # test step
        trainer = pl.Trainer(max_epochs=1)
        trainer.test(model, test_dataloaders=val_dataloader)
        # test epoch end
        results = model.evaluator.compute()
        self.assertGreater(results['AP'], 41.5)
        self.assertGreater(results['AP50'], 62.0)

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
