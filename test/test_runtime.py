# Copyright (c) 2021, yolort team. All Rights Reserved.
from pathlib import Path

import pytest
import torch
from torch import Tensor
from yolort.runtime.trt_helper import YOLOTRTModule
from yolort.v5 import attempt_download


@pytest.mark.parametrize(
    "arch, version, upstream_version, hash_prefix",
    [
        ("yolov5s", "r4.0", "v4.0", "9ca9a642"),
        ("yolov5n", "r6.0", "v6.0", "649e089f"),
        ("yolov5s", "r6.0", "v6.0", "c3b140f3"),
        ("yolov5n6", "r6.0", "v6.0", "beecbbae"),
    ],
)
def test_yolo_trt_module(arch, version, upstream_version, hash_prefix):

    base_url = "https://github.com/ultralytics/yolov5/releases/download/"
    model_url = f"{base_url}/{upstream_version}/{arch}.pt"
    checkpoint_path = attempt_download(model_url, hash_prefix=hash_prefix)

    model = YOLOTRTModule(checkpoint_path, version=version)
    model.eval()
    samples = torch.rand(1, 3, 320, 320)
    outs = model(samples)

    assert isinstance(outs, tuple)
    assert len(outs) == 2
    assert isinstance(outs[0], Tensor)
    assert isinstance(outs[1], Tensor)


@pytest.mark.parametrize(
    "arch, version, upstream_version, hash_prefix",
    [
        ("yolov5s", "r4.0", "v4.0", "9ca9a642"),
        ("yolov5n", "r6.0", "v6.0", "649e089f"),
        ("yolov5s", "r6.0", "v6.0", "c3b140f3"),
        ("yolov5n6", "r6.0", "v6.0", "beecbbae"),
    ],
)
def test_trt_model_onnx_saves(arch, version, upstream_version, hash_prefix):
    base_url = "https://github.com/ultralytics/yolov5/releases/download/"
    model_url = f"{base_url}/{upstream_version}/{arch}.pt"
    checkpoint_path = attempt_download(model_url, hash_prefix=hash_prefix)

    model = YOLOTRTModule(checkpoint_path, version=version)
    model.eval()
    onnx_file_path = f"trt_model_onnx_saves_{arch}_{hash_prefix}.onnx"
    assert not Path(onnx_file_path).exists()
    model.to_onnx(onnx_file_path)
    assert Path(onnx_file_path).exists()
