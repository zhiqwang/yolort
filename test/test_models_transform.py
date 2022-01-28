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
@pytest.mark.parametrize("auto", [True, False])
@pytest.mark.parametrize("stride", [32, 64])
def test_letterbox(img_h, img_w, auto, stride):
    from yolort.models.transform import _letterbox as letterbox1
    from yolort.v5 import letterbox as letterbox2

    img_tensor = torch.randint(0, 255, (3, img_h, img_w))
    img_numpy = img_tensor.permute(1, 2, 0).numpy().astype("uint8")

    out1 = letterbox1(img_tensor, auto=auto, stride=stride)
    out2 = letterbox2(img_numpy, auto=auto, stride=stride)

    assert out1[1] == out2[1]
    assert out1[2] == out2[2]

    aug1 = out1[0][0]
    aug2 = out2[0].astype(float)  # uint8 to float32
    aug2 = np.transpose(aug2 / 255.0, [2, 0, 1])
    aug2 = torch.from_numpy(aug2)
    assert aug1.shape == aug2.shape
    torch.testing.assert_allclose(aug1, aug2, rtol=1e-4, atol=1e-2)
