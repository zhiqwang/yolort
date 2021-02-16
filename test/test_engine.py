import unittest
import torch
import pytorch_lightning as pl

from yolort.models.yolo import yolov5_darknet_pan_s_r31
from yolort.models.transform import nested_tensor_from_tensor_list
from yolort.models import yolov5s

from yolort.datasets import DetectionDataModule

from .dataset_utils import DummyCOCODetectionDataset

from typing import Dict

from torchvision.io import read_image


def default_loader(img_name, is_half=False):
    """
    Read Image using TorchVision.io Here
    """
    img = read_image(img_name)
    img = img.half() if is_half else img.float()  # uint8 to fp16/32
    img /= 255.  # 0 - 255 to 0.0 - 1.0

    return img


class EngineTester(unittest.TestCase):
    def test_train(self):
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
        self.assertIsInstance(out["cls_logits"], torch.Tensor)
        self.assertIsInstance(out["bbox_regression"], torch.Tensor)
        self.assertIsInstance(out["objectness"], torch.Tensor)

    def test_train_one_step(self):
        # Load model
        model = yolov5s()
        model.train()
        # Setup the DataModule
        train_dataset = DummyCOCODetectionDataset(num_samples=128)
        datamodule = DetectionDataModule(train_dataset, batch_size=16)
        # Trainer
        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(model, datamodule)

    def test_inference(self):
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
        self.assertIsInstance(out[0]["boxes"], torch.Tensor)
        self.assertIsInstance(out[0]["labels"], torch.Tensor)
        self.assertIsInstance(out[0]["scores"], torch.Tensor)

    def test_predict_tensor(self):
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
        self.assertIsInstance(predictions[0]["boxes"], torch.Tensor)
        self.assertIsInstance(predictions[0]["labels"], torch.Tensor)
        self.assertIsInstance(predictions[0]["scores"], torch.Tensor)

    def test_predict_tensors(self):
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
        self.assertIsInstance(predictions[0]["boxes"], torch.Tensor)
        self.assertIsInstance(predictions[0]["labels"], torch.Tensor)
        self.assertIsInstance(predictions[0]["scores"], torch.Tensor)

    def test_predict_image_file(self):
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
        self.assertIsInstance(predictions[0]["boxes"], torch.Tensor)
        self.assertIsInstance(predictions[0]["labels"], torch.Tensor)
        self.assertIsInstance(predictions[0]["scores"], torch.Tensor)

    def test_predict_image_files(self):
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
        self.assertIsInstance(predictions[0]["boxes"], torch.Tensor)
        self.assertIsInstance(predictions[0]["labels"], torch.Tensor)
        self.assertIsInstance(predictions[0]["scores"], torch.Tensor)


if __name__ == '__main__':
    unittest.main()
