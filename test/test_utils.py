# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import numpy as np
import pytest
import torch
from torch import Tensor
from yolort import models
from yolort.models import YOLOv5
from yolort.utils import (
    FeatureExtractor,
    get_image_from_url,
    load_from_ultralytics,
    read_image_to_tensor,
)
from yolort.utils.image_utils import box_cxcywh_to_xyxy
from yolort.v5 import letterbox, scale_coords


@pytest.mark.parametrize(
    "arch, version, upstream_version, hash_prefix, use_p6",
    [
        ("yolov5s", "r4.0", "v4.0", "9ca9a642", False),
        ("yolov5s", "r4.0", "v6.0", "c3b140f3", False),
    ],
)
def test_load_from_ultralytics(
    arch: str,
    version: str,
    upstream_version: str,
    hash_prefix: str,
    use_p6: bool,
):
    checkpoint_path = f"{arch}_{version}_{hash_prefix}"
    base_url = "https://github.com/ultralytics/yolov5/releases/download/"
    model_url = f"{base_url}/{upstream_version}/{arch}.pt"

    torch.hub.download_url_to_file(
        model_url,
        checkpoint_path,
        hash_prefix=hash_prefix,
    )
    model_info = load_from_ultralytics(checkpoint_path, version=version)
    assert isinstance(model_info, dict)
    assert model_info["num_classes"] == 80
    assert model_info["size"] == arch.replace("yolov5", "")
    assert model_info["use_p6"] == use_p6
    assert len(model_info["strides"]) == 4 if use_p6 else 3


@pytest.mark.parametrize(
    "arch, version, upstream_version, hash_prefix",
    [("yolov5s", "r4.0", "v4.0", "9ca9a642")],
)
def test_load_from_yolov5(
    arch: str,
    version: str,
    upstream_version: str,
    hash_prefix: str,
):
    img_path = "test/assets/bus.jpg"
    checkpoint_path = f"{arch}_{version}_{hash_prefix}"

    base_url = "https://github.com/ultralytics/yolov5/releases/download/"
    model_url = f"{base_url}/{upstream_version}/{arch}.pt"

    torch.hub.download_url_to_file(
        model_url,
        checkpoint_path,
        hash_prefix=hash_prefix,
    )

    model_yolov5 = YOLOv5.load_from_yolov5(checkpoint_path, version=version)
    model_yolov5.eval()
    out_from_yolov5 = model_yolov5.predict(img_path)
    assert isinstance(out_from_yolov5[0], dict)
    assert isinstance(out_from_yolov5[0]["boxes"], Tensor)
    assert isinstance(out_from_yolov5[0]["labels"], Tensor)
    assert isinstance(out_from_yolov5[0]["scores"], Tensor)

    model = models.__dict__[arch](pretrained=True, score_thresh=0.25)
    model.eval()
    out = model.predict(img_path)

    torch.testing.assert_close(
        out_from_yolov5[0]["scores"], out[0]["scores"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        out_from_yolov5[0]["labels"], out[0]["labels"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        out_from_yolov5[0]["boxes"], out[0]["boxes"], rtol=0, atol=0
    )


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


@pytest.mark.parametrize(
    "batch_size, height, width",
    [
        (8, 640, 640),
        (4, 416, 320),
        (8, 320, 416),
    ]
)
@pytest.mark.parametrize("arch", ["yolov5n", "yolov5s"])
def test_feature_extractor(batch_size, height, width, arch):
    c = 3
    in_channels = [128, 256, 512]
    strides = [8, 16, 32]
    num_outputs = 85
    expected_features = [
        (batch_size, inc, height // s, width // s) for inc, s in zip(in_channels, strides)
    ]
    expected_head_outputs = [
        (batch_size, c, height // s, width // s, num_outputs) for s in strides
    ]

    model = models.__dict__[arch]()
    model = model.train()
    yolo_features = FeatureExtractor(model.model, return_layers=["backbone", "head"])
    images = torch.rand(batch_size, c, height, width)
    targets = torch.rand(61, 6)
    intermediate_features = yolo_features(images, targets)
    features = intermediate_features["backbone"]
    head_outputs = intermediate_features["head"]
    assert isinstance(features, list)
    assert [f.shape for f in features] == expected_features
    assert isinstance(head_outputs, list)
    assert [h.shape for h in head_outputs] == expected_head_outputs
