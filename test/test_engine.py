import unittest
import torch
import pytorch_lightning as pl

from yolort.models import YOLOLitWrapper
from yolort.models.yolo import yolov5_darknet_pan_s_r31
from yolort.models.transform import nested_tensor_from_tensor_list
from yolort.datasets import DetectionDataModule

from .torch_utils import image_preprocess
from .dataset_utils import DummyCOCODetectionDataset

from typing import Dict


class EngineTester(unittest.TestCase):
    def test_train(self):
        # Read Image using TorchVision.io Here
        # Do forward over image
        img_name = "test/assets/zidane.jpg"
        img_tensor = image_preprocess(img_name)
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
        self.assertIsInstance(out["cls_logits"], torch.Tensor)
        self.assertIsInstance(out["bbox_regression"], torch.Tensor)
        self.assertIsInstance(out["objectness"], torch.Tensor)

    def test_train_one_step(self):
        # Load model
        model = YOLOLitWrapper()
        model.train()

        # Setup the DataModule
        train_dataset = DummyCOCODetectionDataset(num_samples=128)
        datamodule = DetectionDataModule(train_dataset, batch_size=16)
        # Trainer
        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(model, datamodule)

    def test_inference(self):
        # Infer over an image
        img_name = "test/assets/zidane.jpg"
        img_input = image_preprocess(img_name)
        self.assertEqual(img_input.ndim, 3)

        model = YOLOLitWrapper(pretrained=True)
        model.eval()

        out = model([img_input])
        self.assertIsInstance(out, list)
        self.assertIsInstance(out[0], Dict)
        self.assertIsInstance(out[0]["boxes"], torch.Tensor)
        self.assertIsInstance(out[0]["labels"], torch.Tensor)
        self.assertIsInstance(out[0]["scores"], torch.Tensor)


if __name__ == '__main__':
    unittest.main()
