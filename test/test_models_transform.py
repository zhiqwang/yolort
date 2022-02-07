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


@pytest.mark.parametrize("img_h", [300, 500, 720, 800, 1080, 1280])
@pytest.mark.parametrize("img_w", [300, 500, 720, 800, 1080, 1280])
@pytest.mark.parametrize("fixed_shape", [True, False])
@pytest.mark.parametrize("stride", [32, 64])
def test_letterbox(img_h, img_w, fixed_shape, stride):

    from yolort.models.transform import _resize_image_and_masks
    from yolort.v5 import letterbox

    new_shape = (640, 640)  # height, width

    img_tensor = torch.randint(0, 255, (3, img_h, img_w))
    img_numpy = img_tensor.permute(1, 2, 0).numpy().astype("uint8")

    yolo_transform = YOLOTransform(
        new_shape[0],
        new_shape[1],
        size_divisible=stride,
        fixed_shape=fixed_shape,
    )

    im3 = img_tensor / 255
    im3, _ = _resize_image_and_masks(im3.float(), new_shape)
    out1 = yolo_transform.batch_images([im3])

    out2 = letterbox(img_numpy, new_shape=new_shape, auto=not fixed_shape, stride=stride)

    aug1 = out1[0].numpy()
    aug2 = out2[0].astype(np.float32)  # uint8 to float32
    aug2 = np.transpose(aug2 / 255.0, [2, 0, 1])
    assert aug1.shape == aug2.shape
    np.testing.assert_allclose(aug1, aug2, rtol=1e-4, atol=1e-2)
