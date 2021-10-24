import copy

import torch
from yolort.models.transform import YOLOTransform, NestedTensor


def test_yolo_transform():
    transform = YOLOTransform(300, 500)
    images = [torch.rand(3, 200, 300), torch.rand(3, 200, 200)]
    annotations = [
        {"boxes": torch.randint(180, (3, 4)), "labels": torch.randint(80, (3,))},
        {"boxes": torch.randint(180, (2, 4)), "labels": torch.randint(80, (2,))},
    ]
    annotations_copy = copy.deepcopy(annotations)
    samples, targets = transform(images, annotations)
    assert isinstance(samples, NestedTensor)
    assert targets.shape[1] == 6

    # Test annotations after transformation
    torch.testing.assert_close(annotations[0]["boxes"], annotations_copy[0]["boxes"], rtol=0, atol=0)
    torch.testing.assert_close(annotations[1]["boxes"], annotations_copy[1]["boxes"], rtol=0, atol=0)
