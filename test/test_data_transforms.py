# Copyright (c) 2022, yolort team. All rights reserved.
import cv2
import torch
import torchvision


def test_letterbox():
    from yolort.data.transforms import letterbox as letterbox1
    from yolort.utils import read_image_to_tensor
    from yolort.v5 import letterbox as letterbox2

    img_source = "test/assets/bus.jpg"
    im1 = torchvision.io.read_image(img_source)
    out1 = letterbox1(im1, auto=False)[0]

    im2 = cv2.imread(img_source)
    im2 = letterbox2(im2, auto=False)[0]
    out2 = read_image_to_tensor(im2)[None]

    assert out1.shape == out2.shape
    torch.testing.assert_allclose(out1, out2)
