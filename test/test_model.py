import torch

from torchvision.models.detection.image_list import ImageList
from models.anchor_utils import AnchorGenerator

from collections import OrderedDict

import unittest


class ModelTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)

    def _init_test_anchor_generator(self):
        anchor_sizes = ((128,), (256,), (512,))
        aspect_ratios = (0.5, 1.0, 2.0)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        return anchor_generator

    def get_features(self, images):
        s0, s1 = images.shape[-2:]
        features = [
            ('0', torch.rand(2, 128, s0 // 8, s1 // 8)),
            ('1', torch.rand(2, 256, s0 // 16, s1 // 16)),
            ('2', torch.rand(2, 512, s0 // 32, s1 // 32)),
        ]
        features = OrderedDict(features)
        return features

    def test_anchor_generator(self):
        images = torch.rand(2, 3, 416, 320)
        features = self.get_features(images)
        features = list(features.values())
        image_shapes = [i.shape[-2:] for i in images]
        images = ImageList(images, image_shapes)

        model = self._init_test_anchor_generator()
        model.eval()
        anchors = model(images, features)
        self.assertEqual(anchors[0].shape[0], 8190)


if __name__ == '__main__':
    unittest.main()
