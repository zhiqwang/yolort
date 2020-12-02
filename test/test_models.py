import torch

from models.anchor_utils import AnchorGenerator
from .common_utils import TestCase


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
