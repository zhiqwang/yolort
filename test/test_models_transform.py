# Copyright (c) 2022, yolort team. All rights reserved.
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


def test_letterbox():
    import cv2
    from torchvision.io import read_image
    from yolort.models.transform import letterbox as letterbox1
    from yolort.utils import read_image_to_tensor
    from yolort.v5 import letterbox as letterbox2

    img_source = "test/assets/bus.jpg"
    im1 = read_image(img_source)
    out1 = letterbox1(im1, auto=False)[0]

    im2 = cv2.imread(img_source)
    im2 = letterbox2(im2, auto=False)[0]
    out2 = read_image_to_tensor(im2)[None]

    assert out1.shape == out2.shape
    torch.testing.assert_allclose(out1, out2, rtol=1e-3, atol=1e-2)
