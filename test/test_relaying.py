import pytest
import torch
from torch import Tensor
from torch.jit._trace import TopLevelTracedModule
from yolort.models import yolov5s
from yolort.relaying import get_trace_module, YOLOInference


def test_get_trace_module():
    model_func = yolov5s(pretrained=True)
    script_module = get_trace_module(model_func, input_shape=(416, 320))
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
def test_yolo_inference(arch, version, upstream_version, hash_prefix):
    checkpoint_path = f"{arch}_{upstream_version}_{hash_prefix}"

    base_url = "https://github.com/ultralytics/yolov5/releases/download/"
    model_url = f"{base_url}/{upstream_version}/{arch}.pt"

    torch.hub.download_url_to_file(
        model_url,
        checkpoint_path,
        hash_prefix=hash_prefix,
    )
    score_thresh = 0.25

    model = YOLOInference(
        checkpoint_path,
        score_thresh=score_thresh,
        version=version,
    )
    model.eval()
    samples = torch.rand(1, 3, 320, 320)
    outs = model(samples)

    assert isinstance(outs[0], dict)
    assert isinstance(outs[0]["boxes"], Tensor)
    assert isinstance(outs[0]["labels"], Tensor)
    assert isinstance(outs[0]["scores"], Tensor)
