import copy
import torch

from yolort.models.transform import YOLOTransform, NestedTensor
from yolort.models._utils import BalancedPositiveNegativeSampler


def test_balanced_positive_negative_sampler():
    sampler = BalancedPositiveNegativeSampler(4, 0.25)
    # keep all 6 negatives first, then add 3 positives, last two are ignore
    matched_idxs = [torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1])]
    pos, neg = sampler(matched_idxs)
    # we know the number of elements that should be sampled for the positive (1)
    # and the negative (3), and their location. Let's make sure that they are there
    assert pos[0].sum() == 1
    assert pos[0][6:9].sum() == 1
    assert neg[0].sum() == 3
    assert neg[0][0:6].sum() == 3


def test_yolo_transform():
    transform = YOLOTransform(300, 500)
    images = [torch.rand(3, 200, 300), torch.rand(3, 200, 200)]
    annotations = [
        {'boxes': torch.randint(180, (3, 4)), 'labels': torch.randint(80, (3,))},
        {'boxes': torch.randint(180, (2, 4)), 'labels': torch.randint(80, (2,))},
    ]
    annotations_copy = copy.deepcopy(annotations)
    samples, targets = transform(images, annotations)
    assert isinstance(samples, NestedTensor)
    assert targets.shape[1] == 6

    # Test annotations after transformation
    torch.testing.assert_allclose(annotations[0]['boxes'], annotations_copy[0]['boxes'], rtol=0., atol=0.)
    torch.testing.assert_allclose(annotations[1]['boxes'], annotations_copy[1]['boxes'], rtol=0., atol=0.)
