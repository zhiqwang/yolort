import unittest
import copy

import torch
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from models import _utils as det_utils


class UtilsTester(unittest.TestCase):
    def test_balanced_positive_negative_sampler(self):
        sampler = det_utils.BalancedPositiveNegativeSampler(4, 0.25)
        # keep all 6 negatives first, then add 3 positives, last two are ignore
        matched_idxs = [torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1])]
        pos, neg = sampler(matched_idxs)
        # we know the number of elements that should be sampled for the positive (1)
        # and the negative (3), and their location. Let's make sure that they are there
        self.assertEqual(pos[0].sum(), 1)
        self.assertEqual(pos[0][6:9].sum(), 1)
        self.assertEqual(neg[0].sum(), 3)
        self.assertEqual(neg[0][0:6].sum(), 3)

    def test_transform_copy_targets(self):
        transform = GeneralizedRCNNTransform(300, 500, torch.zeros(3), torch.ones(3))
        image = [torch.rand(3, 200, 300), torch.rand(3, 200, 200)]
        targets = [{'boxes': torch.rand(3, 4)}, {'boxes': torch.rand(2, 4)}]
        targets_copy = copy.deepcopy(targets)
        out = transform(image, targets)  # noqa: F841
        self.assertTrue(torch.equal(targets[0]['boxes'], targets_copy[0]['boxes']))
        self.assertTrue(torch.equal(targets[1]['boxes'], targets_copy[1]['boxes']))


if __name__ == '__main__':
    unittest.main()
