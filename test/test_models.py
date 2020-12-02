import unittest

import torch
from torchvision.models.detection.image_list import ImageList

from models.anchor_utils import AnchorGenerator
from .common_utils import TestCase


class ModelTester(TestCase):
    def _init_test_anchor_generator(self):
        anchor_sizes = ((10,),)
        aspect_ratios = ((1,),)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        return anchor_generator

    def get_features(self, images):
        s0, s1 = images.shape[-2:]
        features = [torch.rand(2, 8, s0 // 5, s1 // 5)]
        return features

    @unittest.skip("Current it isn't well implemented")
    def test_anchor_generator(self):
        images = torch.randn(2, 3, 15, 15)
        features = self.get_features(images)
        image_shapes = [i.shape[-2:] for i in images]
        images = ImageList(images, image_shapes)

        model = self._init_test_anchor_generator()
        model.eval()
        anchors = model(images, features)

        # Estimate the number of target anchors
        grid_sizes = [f.shape[-2:] for f in features]
        num_anchors_estimated = 0
        for sizes, num_anchors_per_loc in zip(grid_sizes, model.num_anchors_per_location()):
            num_anchors_estimated += sizes[0] * sizes[1] * num_anchors_per_loc

        anchors_output = torch.tensor([[-5., -5., 5., 5.],
                                       [0., -5., 10., 5.],
                                       [5., -5., 15., 5.],
                                       [-5., 0., 5., 10.],
                                       [0., 0., 10., 10.],
                                       [5., 0., 15., 10.],
                                       [-5., 5., 5., 15.],
                                       [0., 5., 10., 15.],
                                       [5., 5., 15., 15.]])

        self.assertEqual(num_anchors_estimated, 9)
        self.assertEqual(len(anchors), 2)
        self.assertEqual(tuple(anchors[0].shape), (9, 4))
        self.assertEqual(tuple(anchors[1].shape), (9, 4))
        self.assertEqual(anchors[0], anchors_output)
        self.assertEqual(anchors[1], anchors_output)
