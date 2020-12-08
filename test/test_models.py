import unittest
import torch

from models.anchor_utils import AnchorGenerator
from models import yolov5s
from .common_utils import TestCase
from .torch_utils import image_preprocess

from typing import Dict


class ModelTester(TestCase):
    def _init_test_anchor_generator(self):
        strides = [4]
        anchor_grids = [[6, 14]]
        anchor_generator = AnchorGenerator(strides, anchor_grids)

        return anchor_generator

    def get_features(self, images):
        s0, s1 = images.shape[-2:]
        features = [torch.rand(2, 8, s0 // 5, s1 // 5)]
        return features

    def test_anchor_generator(self):
        images = torch.randn(2, 3, 10, 10)
        features = self.get_features(images)

        model = self._init_test_anchor_generator()
        model.eval()
        anchors = model(features)

        anchor_output = torch.tensor([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
        wh_output = torch.tensor([[4.], [4.], [4.], [4.]])
        xy_output = torch.tensor([[6., 14.], [6., 14.], [6., 14.], [6., 14.]])

        self.assertEqual(len(anchors), 3)
        self.assertEqual(tuple(anchors[0].shape), (4, 2))
        self.assertEqual(tuple(anchors[1].shape), (4, 1))
        self.assertEqual(tuple(anchors[2].shape), (4, 2))
        self.assertEqual(anchors[0], anchor_output)
        self.assertEqual(anchors[1], wh_output)
        self.assertEqual(anchors[2], xy_output)


class EngineTester(TestCase):
    @unittest.skip("Current it isn't well implemented")
    def test_train(self):
        # Read Image using TorchVision.io Here
        # Do forward over image
        img_name = "test/assets/zidane.jpg"
        img_tensor = image_preprocess(img_name)
        self.assertEqual(img_tensor.ndim, 3)

        boxes = torch.tensor([[0, 0, 100, 100],
                              [0, 12, 25, 225],
                              [10, 15, 30, 35],
                              [23, 35, 93, 95]], dtype=torch.float)
        labels = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        targets = [{"boxes": boxes, "labels": labels}]

        model = yolov5s(num_classes=5)
        out = model(img_tensor, targets)
        self.assertIsInstance(out, Dict)
        self.assertIsInstance(out["loss_classifier"], torch.Tensor)
        self.assertIsInstance(out["loss_box_reg"], torch.Tensor)
        self.assertIsInstance(out["loss_objectness"], torch.Tensor)

    def test_inference(self):
        # Infer over an image
        img_name = "test/assets/zidane.jpg"
        img_input = image_preprocess(img_name)
        self.assertEqual(img_input.ndim, 3)

        model = yolov5s(pretrained=True)
        model.eval()

        out = model([img_input])
        self.assertIsInstance(out, list)
        self.assertIsInstance(out[0], Dict)
        self.assertIsInstance(out[0]["boxes"], torch.Tensor)
        self.assertIsInstance(out[0]["labels"], torch.Tensor)
        self.assertIsInstance(out[0]["scores"], torch.Tensor)
