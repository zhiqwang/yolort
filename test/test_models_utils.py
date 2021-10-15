# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
import pytest
import torch
from torch import Tensor

from yolort import models
from yolort.models import YOLOv5
from yolort.models._utils import load_from_ultralytics


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
