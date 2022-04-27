# Copyright (c) 2021, yolort team. All rights reserved.

from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.jit._trace import TopLevelTracedModule
from yolort.models import yolov5s
from yolort.relay import get_trace_module, YOLOTRTInference
from yolort.relay.trt_graphsurgeon import YOLOTRTGraphSurgeon
from yolort.v5 import attempt_download


@pytest.mark.parametrize("h", [320, 416, 640])
@pytest.mark.parametrize("w", [320, 416, 640])
def test_get_trace_module(h, w):
    model_func = yolov5s(pretrained=True)
    script_module = get_trace_module(model_func, input_shape=(h, w))
    assert isinstance(script_module, TopLevelTracedModule)
    assert script_module.code is not None


@pytest.mark.parametrize(
    "arch, version, upstream_version, hash_prefix",
    [
        ("yolov5s", "r4.0", "v4.0", "9ca9a642"),
        ("yolov5n", "r6.0", "v6.0", "649e089f"),
        ("yolov5s", "r6.0", "v6.0", "c3b140f3"),
        ("yolov5n6", "r6.0", "v6.0", "beecbbae"),
    ],
)
def test_yolo_trt_inference(arch, version, upstream_version, hash_prefix):

    base_url = "https://github.com/ultralytics/yolov5/releases/download/"
    model_url = f"{base_url}/{upstream_version}/{arch}.pt"
    checkpoint_path = attempt_download(model_url, hash_prefix=hash_prefix)

    model = YOLOTRTInference(checkpoint_path, version=version)
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
def test_yolo_trt_inference_to_onnx(arch, version, upstream_version, hash_prefix):
    base_url = "https://github.com/ultralytics/yolov5/releases/download/"
    model_url = f"{base_url}/{upstream_version}/{arch}.pt"
    checkpoint_path = attempt_download(model_url, hash_prefix=hash_prefix)

    model = YOLOTRTInference(checkpoint_path, version=version)
    model.eval()
    onnx_file_path = f"yolo_trt_inference_to_onnx_{arch}_{hash_prefix}.onnx"
    assert not Path(onnx_file_path).exists()
    model.to_onnx(onnx_file_path)
    assert Path(onnx_file_path).exists()


@pytest.mark.parametrize(
    "arch, version, upstream_version, hash_prefix",
    [
        ("yolov5s", "r4.0", "v4.0", "9ca9a642"),
        ("yolov5n", "r6.0", "v6.0", "649e089f"),
        ("yolov5s", "r6.0", "v6.0", "c3b140f3"),
        ("yolov5n6", "r6.0", "v6.0", "beecbbae"),
    ],
)
def test_yolo_graphsurgeon_wo_nms(arch, version, upstream_version, hash_prefix):
    base_url = "https://github.com/ultralytics/yolov5/releases/download/"
    model_url = f"{base_url}/{upstream_version}/{arch}.pt"
    checkpoint_path = attempt_download(model_url, hash_prefix=hash_prefix)

    yolo_gs = YOLOTRTGraphSurgeon(checkpoint_path, version=version, enable_dynamic=False)
    onnx_file_path = f"yolo_graphsurgeon_wo_nms_{arch}_{hash_prefix}.onnx"
    assert not Path(onnx_file_path).exists()
    yolo_gs.save(onnx_file_path)
    assert Path(onnx_file_path).exists()


@pytest.mark.parametrize(
    "arch, version, upstream_version, hash_prefix",
    [
        ("yolov5s", "r4.0", "v4.0", "9ca9a642"),
        ("yolov5n", "r6.0", "v6.0", "649e089f"),
        ("yolov5s", "r6.0", "v6.0", "c3b140f3"),
        ("yolov5n6", "r6.0", "v6.0", "beecbbae"),
    ],
)
def test_yolo_graphsurgeon_register_nms(arch, version, upstream_version, hash_prefix):
    base_url = "https://github.com/ultralytics/yolov5/releases/download/"
    model_url = f"{base_url}/{upstream_version}/{arch}.pt"
    checkpoint_path = attempt_download(model_url, hash_prefix=hash_prefix)

    yolo_gs = YOLOTRTGraphSurgeon(checkpoint_path, version=version, enable_dynamic=False)
    yolo_gs.register_nms()
    onnx_file_path = f"yolo_graphsurgeon_register_nms{arch}_{hash_prefix}.onnx"
    assert not Path(onnx_file_path).exists()
    yolo_gs.save(onnx_file_path)
    assert Path(onnx_file_path).exists()
