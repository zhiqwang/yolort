# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn, Tensor
from yolort.models import yolov5s
from yolort.utils import (
    FeatureExtractor,
    update_module_state_from_ultralytics,
    read_image_to_tensor,
    get_image_from_url,
)
from yolort.utils.image_utils import box_cxcywh_to_xyxy
from yolort.v5 import letterbox, scale_coords


def test_update_module_state_from_ultralytics():
    yolov5s_r40_path = Path("yolov5s.pt")

    if not yolov5s_r40_path.exists():
        torch.hub.download_url_to_file(
            "https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5s.pt",
            yolov5s_r40_path,
            hash_prefix="9ca9a642",
        )

    model = update_module_state_from_ultralytics(
        str(yolov5s_r40_path),
        arch="yolov5s",
        feature_fusion_type="PAN",
        num_classes=80,
    )
    assert isinstance(model, nn.Module)


def test_read_image_to_tensor():
    N, H, W = 3, 720, 360
    img = np.random.randint(0, 255, (H, W, N), dtype="uint8")  # As a dummy image
    out = read_image_to_tensor(img)

    assert isinstance(out, Tensor)
    assert tuple(out.shape) == (N, H, W)


def test_get_image_from_url():
    url = "https://raw.githubusercontent.com/zhiqwang/yolov5-rt-stack/master/test/assets/zidane.jpg"
    img = get_image_from_url(url)
    assert isinstance(img, np.ndarray)
    assert tuple(img.shape) == (720, 1280, 3)


def test_letterbox():
    img = np.random.randint(0, 255, (720, 360, 3), dtype="uint8")  # As a dummy image
    out = letterbox(img, new_shape=(416, 416))[0]
    assert tuple(out.shape) == (416, 224, 3)


def test_box_cxcywh_to_xyxy():
    box_cxcywh = np.asarray(
        [[50, 50, 100, 100], [0, 0, 0, 0], [20, 25, 20, 20], [58, 65, 70, 60]],
        dtype=np.float,
    )
    exp_xyxy = np.asarray(
        [[0, 0, 100, 100], [0, 0, 0, 0], [10, 15, 30, 35], [23, 35, 93, 95]],
        dtype=np.float,
    )

    box_xyxy = box_cxcywh_to_xyxy(box_cxcywh)
    assert exp_xyxy.shape == (4, 4)
    assert exp_xyxy.dtype == box_xyxy.dtype
    np.testing.assert_array_equal(exp_xyxy, box_xyxy)


def test_scale_coords():
    box_tensor = torch.tensor(
        [
            [0.0, 0.0, 100.0, 100.0],
            [0.0, 0.0, 0.0, 0.0],
            [10.0, 15.0, 30.0, 35.0],
            [20.0, 35.0, 90.0, 95.0],
        ],
        dtype=torch.float,
    )
    exp_coords = torch.tensor(
        [
            [0.0, 0.0, 108.05, 111.25],
            [0.0, 0.0, 0.0, 0.0],
            [7.9250, 16.6875, 30.1750, 38.9375],
            [19.05, 38.9375, 96.9250, 105.6875],
        ],
        dtype=torch.float,
    )

    box_coords_scaled = scale_coords((160, 128), box_tensor, (178, 136))
    assert tuple(box_coords_scaled.shape) == (4, 4)
    torch.testing.assert_close(box_coords_scaled, exp_coords)


@pytest.mark.parametrize("b, h, w", [(8, 640, 640), (4, 416, 320), (8, 320, 416)])
def test_feature_extractor(b, h, w):
    c = 3
    in_channels = [128, 256, 512]
    strides = [8, 16, 32]
    num_outputs = 85
    expected_features = [
        (b, inc, h // s, w // s) for inc, s in zip(in_channels, strides)
    ]
    expected_head_outputs = [(b, c, h // s, w // s, num_outputs) for s in strides]

    model = yolov5s()
    model = model.train()
    yolo_features = FeatureExtractor(model.model, return_layers=["backbone", "head"])
    images = torch.rand(b, c, h, w)
    targets = torch.rand(61, 6)
    intermediate_features = yolo_features(images, targets)
    features = intermediate_features["backbone"]
    head_outputs = intermediate_features["head"]
    assert isinstance(features, list)
    assert [f.shape for f in features] == expected_features
    assert isinstance(head_outputs, list)
    assert [h.shape for h in head_outputs] == expected_head_outputs
