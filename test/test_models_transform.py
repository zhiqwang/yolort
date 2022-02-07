# Copyright (c) 2022, yolort team. All rights reserved.
import copy

import numpy as np
import pytest
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


@pytest.mark.parametrize("im_shape", [(500, 500), (500, 1080), (720, 900), (1000, 950), (900, 720)])
@pytest.mark.parametrize("auto", [True, False])
@pytest.mark.parametrize("stride", [32, 64])
def test_letterbox(im_shape, auto, stride):

    from yolort.models.transform import _resize_image_and_masks
    from yolort.v5 import letterbox

    min_size = 640
    max_size = 640
    new_shape = (min_size, max_size)
    fixed_shape = None if auto else new_shape

    im_tensor = torch.randint(0, 255, (3, *(im_shape)))
    im_numpy = im_tensor.permute(1, 2, 0).numpy().astype("uint8")

    yolo_transform = YOLOTransform(
        min_size,
        max_size,
        size_divisible=stride,
        fixed_shape=fixed_shape,
    )

    im1 = im_tensor / 255
    im1, _ = _resize_image_and_masks(im1, float(min_size), float(max_size))
    out1 = yolo_transform.batch_images([im1])

    out2 = letterbox(im_numpy, new_shape=new_shape, auto=auto, stride=stride)

    aug1 = out1[0].numpy()
    aug2 = out2[0].astype(np.float32)  # uint8 to float32
    aug2 = np.transpose(aug2 / 255.0, [2, 0, 1])
    assert aug1.shape == aug2.shape
    np.testing.assert_allclose(aug1, aug2, rtol=1e-4, atol=1e-2)
